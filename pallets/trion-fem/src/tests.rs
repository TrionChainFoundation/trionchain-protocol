use crate::mock::{new_test_ext, RuntimeOrigin, TrionFem};

#[test]
fn update_cell_works() {
    new_test_ext().execute_with(|| {
        assert!(TrionFem::update_cell(RuntimeOrigin::signed(1), 1, 10, 100, 200).is_ok());
        let stored = TrionFem::cell_states(1).unwrap();
        assert_eq!(stored.step, 10);
        assert_eq!(stored.temperature, 100);
        assert_eq!(stored.pressure, 200);
    });
}
