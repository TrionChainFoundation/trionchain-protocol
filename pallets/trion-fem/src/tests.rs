use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok};

#[test]
fn it_works_for_authorized_sensor() {
	new_test_ext().execute_with(|| {
		// 1. Configurar bloque 1
		System::set_block_number(1);
		
		// 2. Registrar sensor (Alice=1 es dueña de Celda=100)
		assert_ok!(TemplateModule::register_sensor(RuntimeOrigin::signed(1), 100, 1));
		
		// Verificar evento de registro
		System::assert_last_event(Event::SensorAuthorized { cell_id: 100, operator: 1 }.into());

		// 3. Reportar estado válido
		assert_ok!(TemplateModule::report_state(
			RuntimeOrigin::signed(1), 
			100, // Cell ID
			500, // Stress
			100, // Generation
			1,   // Demand
			50,  // Soc
			0    // Price
		));

		// 4. Verificar que se guardó en la base de datos
		let data = TemplateModule::get_cell_state(100).unwrap();
		assert_eq!(data.stress, 500);
		
		// Verificar evento de reporte físico
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
		
		// La cuenta 2 (Bob) intenta reportar en la celda 100 (sin dueño)
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
		
		// Registro correcto
		assert_ok!(TemplateModule::register_sensor(RuntimeOrigin::signed(1), 100, 1));

		// Intento de ataque: Stress 2000 (Límite físico es 1000)
		assert_noop!(
			TemplateModule::report_state(RuntimeOrigin::signed(1), 100, 2000, 100, 1, 50, 0),
			Error::<Test>::InvalidPhysicalValue
		);
	});
}