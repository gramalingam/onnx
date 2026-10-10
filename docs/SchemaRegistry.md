<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Dynamic schema registry lifecycle

ONNX already supports opaque `TypeProto` values. Dynamic operator providers need
registry synchronization and domain lifecycle management independently of the
types their operators accept.

Registration, deregistration, lookup, and schema enumeration synchronize access
to the schema map. Enumeration returns copies. **Lookup returns borrowed schema
pointers, not lifetime handles.** A caller must exclude deregistration for the
entire lifetime of each pointer, including model checking, shape inference, and
execution of schema callbacks. Before unloading a provider, its owner must also
stop users of copied schemas containing callbacks into that provider.
These APIs do not provide concurrent plugin unload safety.

Domain `MapSnapshot()` and `LastReleaseVersionMapSnapshot()` return independent
copies under the domain mutex. Each call is synchronized separately; two calls
are not a transaction. The existing reference-returning `Map()` and
`LastReleaseVersionMap()` remain available for compatibility, but require callers
to exclude concurrent domain mutations. ONNX's internal readers use snapshots.

For a temporary provider, coordinate the following steps externally:

1. Save the domain's original range and last-release version, or record that it
   was absent. Add or widen the range before registering schemas.
2. Register schemas with `RegisterSchema(..., fail_with_exception=true)` so a
   registration failure can be reported and rolled back. On failure, deregister
   only schemas successfully installed by this provider, then perform cleanup.
3. After stopping schema users, deregister the provider's schemas and call
   `OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, had_original_range,
   min_version, max_version, last_release_version)`.

Cleanup checks the schema map and changes domain metadata atomically with respect
to schema and domain mutations. It leaves metadata unchanged if **any** schemas
remain in that domain, rather than narrowing below another registrar's versions.
Otherwise it restores the saved metadata or removes the newly added domain.
Unknown domains and invalid saved ranges raise `SchemaError`.

Providers sharing a domain must share one original baseline and coordinate the
whole lifecycle, including failure rollback and the final cleanup. Cleanup does
not count owners, authenticate the supplied baseline, or schedule deferred
restoration. A stale per-provider baseline is not a safe ownership protocol.
An externally added domain with no live schemas cannot be distinguished from a
temporary domain without explicit retention.

A permanent registrar can call
`DomainToVersionRange::RetainDomainToVersion(domain, min_version, max_version,
last_release_version=-1)` to prevent temporary cleanup from removing or restoring
its domain. This creates the domain if absent and otherwise widens its existing
range and last-release version; it never narrows them. The default last-release
version is the requested maximum. Retention is permanent and idempotent, not
reference-counted. There is no provider-specific default range. As with the
existing domain APIs, custom registrars may interpret the last-release metadata
independently of the supported schema range.

Registry writers take the registry mutex before the domain mutex. First-time
static schema initialization also serializes schema mutations using a reentrant
initialization mutex, so debug schema counts cannot race with dynamic writers.
Schema finalization runs before registry locks are acquired. Multi-call opset
registration, domain setup, and lifecycle rollback are not transactions and
still require external coordination.

Deregistration destroys removed schemas after releasing registry locks, allowing
callback capture destructors to reenter the registry. Schema enumeration copies
schemas while holding a read lock; custom callable copy constructors must not
reenter the registry. Python `get_schema` copies a borrowed pointer after lookup,
so callers must exclude native deregistration during that call too.
