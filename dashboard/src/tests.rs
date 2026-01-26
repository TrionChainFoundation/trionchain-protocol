cat << 'EOF' > pallets/template/src/tests.rs
use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok};

#[test]
fn it_works_for_authorized_sensor() {
	new_test_ext().execute_with(|| {
		// 1. Simular que la Cuenta 1 (Alice) registra la Celda 100
		System::set_block_number(1);
		
		assert_ok!(TemplateModule::register_sensor(RuntimeOrigin::signed(1), 100, 1));
		
		// Verificar evento de registro
		System::assert_last_event(Event::SensorAuthorized { cell_id: 100, operator: 1 }.into());

		// 2. Simular que la Cuenta 1 envía un reporte válido
		assert_ok!(TemplateModule::report_state(
			RuntimeOrigin::signed(1), 
			100, // Cell ID
			500, // Stress (Válido < 1000)
			100, // Generation
			1,   // Demand
			50,  // Soc
			0    // Price
		));

		// 3. Verificar que el dato se guardó en la blockchain
		let data = TemplateModule::get_cell_state(100).unwrap();
		assert_eq!(data.stress, 500);
		
		// Verificar evento de reporte
		System::assert_last_event(Event::CellUpdateReceived { 
			cell_id: 100, 
			who: 1, 
			stress: 500, 
			generation: 100,
			demand: 1,
			soc: 50,
			price: 0
		}.into());
	});
}

#[test]
fn it_fails_if_not_authorized() {
	new_test_ext().execute_with(|| {
		System::set_block_number(1);
		
		// La Cuenta 2 intenta reportar en la Celda 100 (que no tiene dueño registrado)
		assert_noop!(
			TemplateModule::report_state(RuntimeOrigin::signed(2), 100, 500, 100, 1, 50, 0),
			Error::<Test>::SensorNotRegistered
		);
	});
}

#[test]
fn it_rejects_impossible_physics() {
	new_test_ext().execute_with(|| {
		System::set_block_number(1);
		
		// Registramos correctamente
		assert_ok!(TemplateModule::register_sensor(RuntimeOrigin::signed(1), 100, 1));

		// Intentamos enviar Stress 2000 (El límite físico es 1000)
		assert_noop!(
			TemplateModule::report_state(RuntimeOrigin::signed(1), 100, 2000, 100, 1, 50, 0),
			Error::<Test>::InvalidPhysicalValue
		);
	});
}
EOF