// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::Test {

namespace {

std::string LifecycleDomain() {
  static std::atomic<int> next_id{0};
  return "test.schema.lifecycle." + std::to_string(next_id++);
}

void RegisterLifecycleSchema(const std::string& domain, int version = 1) {
  RegisterSchema(OpSchema().SetName("LifecycleOp").SetDomain(domain).SinceVersion(version), 0, true, true);
}

void RemoveLifecycleDomain(const std::string& domain) {
  OpSchemaRegistry::OpSchemaDeregisterAll(domain);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, false, 0, 0, 0);
}

} // namespace

TEST(SchemaRegistrationTest, DisabledOnnxStaticRegistrationAPICall) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  EXPECT_TRUE(IsOnnxStaticRegistrationDisabled());
#else
  EXPECT_FALSE(IsOnnxStaticRegistrationDisabled());
#endif
}

// Run this test on its own in a fresh process to exercise lazy initialization.
TEST(SchemaRegistrationTest, ConcurrentInitialLookupAndRegistration) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  ranges.AddDomainToVersion(domain, 1, 100);
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.emplace_back([&]() {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (int version = 1; version <= 100; ++version) {
      RegisterLifecycleSchema(domain, version);
    }
  });
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      while (!start.load()) {
        std::this_thread::yield();
      }
      const auto* schema = OpSchemaRegistry::Schema("Add", 13);
#ifndef __ONNX_DISABLE_STATIC_REGISTRATION
      EXPECT_NE(schema, nullptr);
      EXPECT_FALSE(OpSchemaRegistry::get_all_schemas_with_history().empty());
#else
      (void)schema;
      (void)OpSchemaRegistry::get_all_schemas_with_history();
#endif
    });
  }
  start = true;
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(OpSchemaRegistry::Schema("LifecycleOp", domain)->SinceVersion(), 100);
  RemoveLifecycleDomain(domain);
}

TEST(SchemaRegistrationTest, DomainSnapshotsAreIndependent) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  ranges.AddDomainToVersion(domain, 1, 4, 3);
  const auto versions = ranges.MapSnapshot();
  const auto releases = ranges.LastReleaseVersionMapSnapshot();
  ranges.UpdateDomainToVersion(domain, 1, 8, 7);
  EXPECT_EQ(versions.at(domain), std::make_pair(1, 4));
  EXPECT_EQ(releases.at(domain), 3);
  EXPECT_EQ(ranges.MapSnapshot().at(domain), std::make_pair(1, 8));
  RemoveLifecycleDomain(domain);
  EXPECT_EQ(versions.at(domain), std::make_pair(1, 4));
  EXPECT_EQ(ranges.MapSnapshot().count(domain), 0);
  EXPECT_EQ(ranges.LastReleaseVersionMapSnapshot().count(domain), 0);
}

TEST(SchemaRegistrationTest, CleanupPreservesSharedLiveVersions) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  ranges.AddDomainToVersion(domain, 2, 4, 1);
  ranges.UpdateDomainToVersion(domain, 1, 9, 8);
  RegisterLifecycleSchema(domain, 1);
  RegisterLifecycleSchema(domain, 9);
  DeregisterSchema("LifecycleOp", 1, domain);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, true, 2, 4, 1);
  EXPECT_EQ(ranges.MapSnapshot().at(domain), std::make_pair(1, 9));
  EXPECT_EQ(ranges.LastReleaseVersionMapSnapshot().at(domain), 8);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, false, 0, 0, 0);
  EXPECT_EQ(ranges.MapSnapshot().count(domain), 1);
  DeregisterSchema("LifecycleOp", 9, domain);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, true, 2, 4, 1);
  EXPECT_EQ(ranges.MapSnapshot().at(domain), std::make_pair(2, 4));
  EXPECT_EQ(ranges.LastReleaseVersionMapSnapshot().at(domain), 1);
  RemoveLifecycleDomain(domain);
}

TEST(SchemaRegistrationTest, RetentionCreatesAndWidensWithoutNarrowing) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto new_domain = LifecycleDomain();
  ranges.RetainDomainToVersion(new_domain, 2, 5);
  EXPECT_EQ(ranges.MapSnapshot().at(new_domain), std::make_pair(2, 5));
  EXPECT_EQ(ranges.LastReleaseVersionMapSnapshot().at(new_domain), 5);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(new_domain, false, 0, 0, 0);
  EXPECT_EQ(ranges.MapSnapshot().count(new_domain), 1);

  const auto existing_domain = LifecycleDomain();
  ranges.AddDomainToVersion(existing_domain, 3, 7, 6);
  ranges.RetainDomainToVersion(existing_domain, 2, 5, 4);
  ranges.RetainDomainToVersion(existing_domain, 4, 9, 8);
  ranges.RetainDomainToVersion(existing_domain, 4, 9, 8);
  RegisterLifecycleSchema(existing_domain, 9);
  DeregisterSchema("LifecycleOp", 9, existing_domain);
  OpSchemaRegistry::RestoreDomainToVersionIfUnused(existing_domain, true, 3, 7, 6);
  EXPECT_EQ(ranges.MapSnapshot().at(existing_domain), std::make_pair(2, 9));
  EXPECT_EQ(ranges.LastReleaseVersionMapSnapshot().at(existing_domain), 8);
}

TEST(SchemaRegistrationTest, RegistrationFailureRollbackAndInvalidCleanup) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  EXPECT_THROW(RegisterLifecycleSchema(domain), SchemaError);
  EXPECT_EQ(OpSchemaRegistry::Schema("LifecycleOp", domain), nullptr);
  EXPECT_THROW(OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, false, 0, 0, 0), SchemaError);
  EXPECT_THROW(ranges.RetainDomainToVersion(domain, 5, 2), SchemaError);
  EXPECT_EQ(ranges.MapSnapshot().count(domain), 0);

  ranges.AddDomainToVersion(domain, 1, 2);
  RegisterLifecycleSchema(domain);
  EXPECT_THROW(RegisterLifecycleSchema(domain), SchemaError);
  EXPECT_THROW(RegisterLifecycleSchema(domain, 3), SchemaError);
  EXPECT_EQ(OpSchemaRegistry::Schema("LifecycleOp", domain)->SinceVersion(), 1);
  EXPECT_THROW(OpSchemaRegistry::RestoreDomainToVersionIfUnused(domain, true, 2, 1, 1), SchemaError);
  EXPECT_EQ(ranges.MapSnapshot().at(domain), std::make_pair(1, 2));
  DeregisterSchema("LifecycleOp", 1, domain);
  EXPECT_THROW(DeregisterSchema("LifecycleOp", 1, domain), SchemaError);
  RemoveLifecycleDomain(domain);
}

TEST(SchemaRegistrationTest, ConcurrentSnapshotsEnumerationAndMutation) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  ranges.AddDomainToVersion(domain, 1, 2);
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.emplace_back([&]() {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 100; ++i) {
      ranges.UpdateDomainToVersion(domain, 1, 2 + i);
      RegisterLifecycleSchema(domain);
      DeregisterSchema("LifecycleOp", 1, domain);
    }
  });
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      while (!start.load()) {
        std::this_thread::yield();
      }
      for (int j = 0; j < 10; ++j) {
        EXPECT_EQ(ranges.MapSnapshot().at(domain).first, 1);
        EXPECT_GE(ranges.LastReleaseVersionMapSnapshot().at(domain), 2);
        // Do not dereference the borrowed pointer: the writer may remove it.
        (void)OpSchemaRegistry::Schema("LifecycleOp", domain);
        (void)OpSchemaRegistry::Schema("LifecycleOp", 1, domain);
        (void)OpSchemaRegistry::get_all_schemas();
        (void)OpSchemaRegistry::get_all_schemas_with_history();
      }
    });
  }
  start = true;
  for (auto& thread : threads) {
    thread.join();
  }
  RemoveLifecycleDomain(domain);
}

TEST(SchemaRegistrationTest, CallbackCaptureDestructionCanReenterRegistry) {
  auto& ranges = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto domain = LifecycleDomain();
  ranges.AddDomainToVersion(domain, 1, 2);
  std::atomic<int> destroyed{0};
  struct Capture {
    explicit Capture(std::atomic<int>& count) : destroyed(count) {}
    std::atomic<int>& destroyed;
    ~Capture() {
      (void)OpSchemaRegistry::DomainToVersionRange::Instance().MapSnapshot();
      (void)OpSchemaRegistry::get_all_schemas();
      ++destroyed;
    }
  };
  for (int version = 1; version <= 2; ++version) {
    auto capture = std::make_shared<Capture>(destroyed);
    auto schema = OpSchema().SetName("LifecycleOp").SetDomain(domain).SinceVersion(version);
    schema.TypeAndShapeInferenceFunction([capture](InferenceContext&) {});
    RegisterSchema(std::move(schema), 0, true, true);
  }
  DeregisterSchema("LifecycleOp", 1, domain);
  EXPECT_EQ(destroyed, 1);
  OpSchemaRegistry::OpSchemaDeregisterAll(domain);
  EXPECT_EQ(destroyed, 2);
  RemoveLifecycleDomain(domain);
}

// Schema of all versions are registered by default
// Further schema manipulation expects to be error-free
TEST(SchemaRegistrationTest, RegisterAllByDefaultAndManipulateSchema) {
#ifndef __ONNX_DISABLE_STATIC_REGISTRATION

  // Expects all opset registered by default
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all versions
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();

  // Should not find any opset
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));

  // Register all opset versions
  RegisterOnnxOperatorSetSchema();

  // Should find all opset
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
#endif
}

// By default ONNX registers all opset versions and selective schema loading cannot be tested
// So these tests are run only when static registration is disabled
TEST(SchemaRegistrationTest, RegisterAndDeregisterAllOpsetSchemaVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find schema for any op
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Register all opset versions
  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all ops. Available versions are:
  // Acos-7
  // Add-1,6,7,13,14
  // Trilu-14
  auto schema = OpSchemaRegistry::Schema("Acos", 7);
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 7);

  schema = OpSchemaRegistry::Schema("Add", 14);
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 14);

  schema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, schema);
  EXPECT_EQ(schema->SinceVersion(), 14);

  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));

  // Clear all opset schema registration
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Should not find schema for any op
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_EQ(nullptr, OpSchemaRegistry::Schema("Trilu"));
#endif
}

TEST(SchemaRegistrationTest, RegisterSpecifiedOpsetSchemaVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);
  RegisterOnnxOperatorSetSchema(13);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 13);

  auto opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 13);

  // Should not find opset 12
  opSchema = OpSchemaRegistry::Schema("Add", 12);
  EXPECT_EQ(nullptr, opSchema);

  // Should not find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_EQ(nullptr, opSchema);

  // Acos-7 is the latest Acos before specified 13
  opSchema = OpSchemaRegistry::Schema("Acos", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);
#endif
}

// Register opset-11, then opset-14
// Expects Reg(11, 14) == Reg(11) U Reg(14)
TEST(SchemaRegistrationTest, RegisterMultipleOpsetSchemaVersionsUpgradeVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 11
  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);
  // Register opset 14
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(14, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);

  // Acos-7 is the latest before/at opset 11 and 14
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Add-7 is the latest before/at opset 11
  // Add-14 is the latest before/at opset 14
  // Should find both Add-7,14
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14
  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Should find Add-7 as the max version <=13
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Should find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);
#endif
}

// Register opset-14, then opset-11
// Expects Reg(14, 11) == Reg(11) U Reg(14)
TEST(SchemaRegistrationTest, RegisterMultipleOpsetSchemaVersionsDowngradeVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 14
  RegisterOnnxOperatorSetSchema(14);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 14);
  // Register opset 11
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Acos-7 is the latest before/at opset 11 and 14
  auto opSchema = OpSchemaRegistry::Schema("Acos");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Add-7 is the latest before/at opset 11
  // Add-14 is the latest before/at opset 14
  // Should find both Add-7,14
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 14));

  // Should find the max version 14
  opSchema = OpSchemaRegistry::Schema("Add");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);

  // Should find Add-7 as the max version <=13
  opSchema = OpSchemaRegistry::Schema("Add", 13);
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 7);

  // Should find opset 14
  opSchema = OpSchemaRegistry::Schema("Trilu");
  EXPECT_NE(nullptr, opSchema);
  EXPECT_EQ(opSchema->SinceVersion(), 14);
#endif
}

// Register opset-11, then all versions
// Expects no error
TEST(SchemaRegistrationTest, RegisterSpecificThenAllVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register opset 11
  RegisterOnnxOperatorSetSchema(11);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Register all opset versions
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(0, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Should find schema for all ops
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));
#endif
}

// Register all versions, then opset 11
// Expects no error
TEST(SchemaRegistrationTest, RegisterAllThenSpecificVersion) {
#ifdef __ONNX_DISABLE_STATIC_REGISTRATION
  DeregisterOnnxOperatorSetSchema();
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), -1);

  // Register all opset versions
  RegisterOnnxOperatorSetSchema(0);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 0);

  // Register opset 11
  // Do not fail on duplicate schema registration request
  RegisterOnnxOperatorSetSchema(11, false);
  EXPECT_EQ(OpSchemaRegistry::Instance()->GetLoadedSchemaVersion(), 11);

  // Should find schema for all ops
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Acos"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add"));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Trilu"));

  // Should find schema for all versions
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 1));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 6));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 7));
  EXPECT_NE(nullptr, OpSchemaRegistry::Schema("Add", 13));
#endif
}

} // namespace ONNX_NAMESPACE::Test
