use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok};
use sp_runtime::DispatchError;

// Test accounts
const ROOT: u64 = 0;
const ORACLE_1: u64 = 1;
const ORACLE_2: u64 = 2;
const UNAUTHORIZED: u64 = 99;

// ============================================================================
// Permission Model Tests (Grant Requirement: Production-Grade Governance)
// ============================================================================

#[test]
fn root_can_set_cell_owner() {
    new_test_ext().execute_with(|| {
        // Root assigns oracle for cell 1
        assert_ok!(TrionFem::set_cell_owner(
            RuntimeOrigin::root(),
            1,
            ORACLE_1
        ));

        // Verify storage
        assert_eq!(TrionFem::cell_owner(1), Some(ORACLE_1));

        // Verify event
        System::assert_last_event(
            Event::CellOwnerSet {
                cell_id: 1,
                owner: ORACLE_1,
            }
            .into(),
        );
    });
}

#[test]
fn non_root_cannot_set_cell_owner() {
    new_test_ext().execute_with(|| {
        // Regular user cannot assign oracle
        assert_noop!(
            TrionFem::set_cell_owner(
                RuntimeOrigin::signed(UNAUTHORIZED),
                1,
                ORACLE_1
            ),
            DispatchError::BadOrigin
        );
    });
}

#[test]
fn root_can_set_neighbors() {
    new_test_ext().execute_with(|| {
        // Root configures mesh topology
        assert_ok!(TrionFem::set_neighbors(
            RuntimeOrigin::root(),
            1,
            vec![2, 3, 4]
        ));

        // Verify storage
        let neighbors = TrionFem::neighbors(1).unwrap();
        assert_eq!(neighbors.to_vec(), vec![2, 3, 4]);

        // Verify event
        System::assert_last_event(
            Event::NeighborsSet {
                cell_id: 1,
                neighbor_count: 3,
            }
            .into(),
        );
    });
}

#[test]
fn non_root_cannot_set_neighbors() {
    new_test_ext().execute_with(|| {
        // Regular user cannot configure mesh topology
        assert_noop!(
            TrionFem::set_neighbors(
                RuntimeOrigin::signed(UNAUTHORIZED),
                1,
                vec![2, 3]
            ),
            DispatchError::BadOrigin
        );
    });
}

#[test]
fn unauthorized_user_cannot_update_cell() {
    new_test_ext().execute_with(|| {
        // Setup: Root assigns oracle
        assert_ok!(TrionFem::set_cell_owner(
            RuntimeOrigin::root(),
            1,
            ORACLE_1
        ));

        // Unauthorized user tries to update
        assert_noop!(
            TrionFem::update_cell(
                RuntimeOrigin::signed(UNAUTHORIZED),
                1,
                0,
                500,
                vec![1, 2, 3]
            ),
            Error::<Test>::NotAuthorized
        );
    });
}

#[test]
fn authorized_oracle_can_update_cell() {
    new_test_ext().execute_with(|| {
        // Setup: Root assigns oracle
        assert_ok!(TrionFem::set_cell_owner(
            RuntimeOrigin::root(),
            1,
            ORACLE_1
        ));

        // Authorized oracle updates cell
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1,
            0,
            500,
            vec![1, 2, 3]
        ));

        // Verify storage
        let cell = TrionFem::cell_state(1).unwrap();
        assert_eq!(cell.co2_level, 500);
        assert_eq!(cell.step, 0);
        assert_eq!(cell.physics.to_vec(), vec![1, 2, 3]);

        // Verify event
        System::assert_last_event(
            Event::CellUpdated {
                cell_id: 1,
                step: 0,
                co2_level: 500,
                physics_len: 3,
            }
            .into(),
        );
    });
}

// ============================================================================
// Spatial Validation Tests (Grant Requirement: Physics Consistency)
// ============================================================================

#[test]
fn spatial_validation_accepts_within_threshold() {
    new_test_ext().execute_with(|| {
        // Setup: Create neighbor cells with CO2 data
        // Cell 2: CO2 = 500
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 2, ORACLE_1));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            2, 0, 500, vec![1]
        ));

        // Cell 3: CO2 = 520
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 3, ORACLE_1));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            3, 0, 520, vec![1]
        ));

        // Cell 4: CO2 = 480
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 4, ORACLE_1));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            4, 0, 480, vec![1]
        ));

        // Configure Cell 1 with neighbors 2, 3, 4
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));
        assert_ok!(TrionFem::set_neighbors(
            RuntimeOrigin::root(),
            1,
            vec![2, 3, 4]
        ));

        // Neighbor average: (500 + 520 + 480) / 3 = 500
        // Update Cell 1 with CO2 = 550
        // Delta = 50, which is < MaxCo2Delta (100) → Should ACCEPT
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1, 1, 550, vec![2]
        ));

        let cell = TrionFem::cell_state(1).unwrap();
        assert_eq!(cell.co2_level, 550);
    });
}

#[test]
fn spatial_validation_rejects_exceeding_threshold() {
    new_test_ext().execute_with(|| {
        // Setup: Create neighbor cells with consistent CO2
        // Cell 2: CO2 = 500
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 2, ORACLE_1));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            2, 0, 500, vec![1]
        ));

        // Cell 3: CO2 = 500
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 3, ORACLE_1));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            3, 0, 500, vec![1]
        ));

        // Configure Cell 1 with neighbors
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));
        assert_ok!(TrionFem::set_neighbors(
            RuntimeOrigin::root(),
            1,
            vec![2, 3]
        ));

        // Neighbor average: (500 + 500) / 2 = 500
        // Try to update with CO2 = 650
        // Delta = 150, which exceeds MaxCo2Delta (100) → Should REJECT
        assert_noop!(
            TrionFem::update_cell(
                RuntimeOrigin::signed(ORACLE_1),
                1, 1, 650, vec![2]
            ),
            Error::<Test>::Co2DeltaTooHigh
        );
    });
}

#[test]
fn spatial_validation_allows_bootstrap_no_neighbor_data() {
    new_test_ext().execute_with(|| {
        // Setup: Cell 1 has neighbors configured, but they have no data yet
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));
        assert_ok!(TrionFem::set_neighbors(
            RuntimeOrigin::root(),
            1,
            vec![2, 3, 4]
        ));

        // Neighbors 2, 3, 4 have no data yet (bootstrap scenario)
        // Should allow ANY CO2 value since we can't validate against neighbors
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1, 0, 999, vec![1]
        ));

        let cell = TrionFem::cell_state(1).unwrap();
        assert_eq!(cell.co2_level, 999);
    });
}

#[test]
fn spatial_validation_allows_no_neighbors_configured() {
    new_test_ext().execute_with(|| {
        // Setup: Cell 1 has no neighbors configured (isolated cell)
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));

        // Should allow any CO2 value since cell is isolated
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1, 0, 999, vec![1]
        ));

        let cell = TrionFem::cell_state(1).unwrap();
        assert_eq!(cell.co2_level, 999);
    });
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

#[test]
fn rejects_too_many_neighbors() {
    new_test_ext().execute_with(|| {
        // Try to set more neighbors than MaxNeighbors (8)
        assert_noop!(
            TrionFem::set_neighbors(
                RuntimeOrigin::root(),
                1,
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9] // 9 neighbors, max is 8
            ),
            Error::<Test>::TooManyNeighbors
        );
    });
}

#[test]
fn rejects_physics_data_too_long() {
    new_test_ext().execute_with(|| {
        // Setup: Assign oracle
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));

        // Create physics data longer than MaxPhysicsLen (256)
        let long_physics = sp_std::vec![1u32; 257];

        assert_noop!(
            TrionFem::update_cell(
                RuntimeOrigin::signed(ORACLE_1),
                1, 0, 500, long_physics
            ),
            Error::<Test>::PhysicsTooLong
        );
    });
}

// ============================================================================
// Integration Tests (Complete Workflow)
// ============================================================================

#[test]
fn complete_workflow_multiple_cells() {
    new_test_ext().execute_with(|| {
        // Step 1: Root configures mesh topology
        // Cell topology: 1 ← → 2 ← → 3 (linear mesh)
        assert_ok!(TrionFem::set_neighbors(RuntimeOrigin::root(), 1, vec![2]));
        assert_ok!(TrionFem::set_neighbors(RuntimeOrigin::root(), 2, vec![1, 3]));
        assert_ok!(TrionFem::set_neighbors(RuntimeOrigin::root(), 3, vec![2]));

        // Step 2: Root assigns oracles
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 2, ORACLE_1));
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 3, ORACLE_2));

        // Step 3: Oracles report initial state (bootstrap - no validation)
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1, 0, 400, vec![1]
        ));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            2, 0, 450, vec![1]
        ));
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_2),
            3, 0, 500, vec![1]
        ));

        // Step 4: Update cell 2 - should validate against neighbors 1 and 3
        // Neighbor avg: (400 + 500) / 2 = 450
        // New value: 480, delta = 30 < 100 → Should ACCEPT
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            2, 1, 480, vec![2]
        ));

        // Step 5: Try physically implausible update
        // Neighbor avg still 450, try 600, delta = 150 > 100 → Should REJECT
        assert_noop!(
            TrionFem::update_cell(
                RuntimeOrigin::signed(ORACLE_1),
                2, 2, 600, vec![3]
            ),
            Error::<Test>::Co2DeltaTooHigh
        );
    });
}

#[test]
fn oracle_transfer_workflow() {
    new_test_ext().execute_with(|| {
        // Step 1: Root assigns initial oracle
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_1));

        // Step 2: Oracle 1 updates cell
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_1),
            1, 0, 500, vec![1]
        ));

        // Step 3: Root transfers oracle to ORACLE_2
        assert_ok!(TrionFem::set_cell_owner(RuntimeOrigin::root(), 1, ORACLE_2));

        // Step 4: Old oracle can no longer update
        assert_noop!(
            TrionFem::update_cell(
                RuntimeOrigin::signed(ORACLE_1),
                1, 1, 510, vec![2]
            ),
            Error::<Test>::NotAuthorized
        );

        // Step 5: New oracle can update
        assert_ok!(TrionFem::update_cell(
            RuntimeOrigin::signed(ORACLE_2),
            1, 1, 510, vec![2]
        ));
    });
}