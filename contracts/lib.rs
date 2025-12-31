#![cfg_attr(not(feature = "std"), no_std, no_main)]

#[ink::contract]
mod phyfi_insurance {

    // 1. STORAGE: ¿Qué datos guardamos?
    #[ink(storage)]
    pub struct Insurance {
        owner: AccountId,           // El dueño del contrato (La Aseguradora)
        insured: AccountId,         // El cliente (Aerolínea)
        stress_threshold: u32,      // Límite físico (ej. 800)
        payout_amount: Balance,     // Cuánto pagar si se rompe
        is_active: bool,            // Si el seguro está vivo
    }

    impl Insurance {
        // 2. CONSTRUCTOR: Configurar el seguro
        #[ink(constructor)]
        pub fn new(insured_client: AccountId, threshold: u32) -> Self {
            let caller = Self::env().caller();
            Self {
                owner: caller,
                insured: insured_client,
                stress_threshold: threshold,
                payout_amount: 0,
                is_active: true,
            }
        }

        // 3. MESSAGES: Funciones públicas

        /// El cliente deposita la prima (dinero) o la aseguradora fondea el contrato
        #[ink(message, payable)]
        pub fn fund_contract(&mut self) {
            self.payout_amount = self.env().transferred_value();
        }

        /// EL ORÁCULO llama a esto con datos físicos
        #[ink(message)]
        pub fn report_physics(&mut self, current_stress: u32) {
            // Solo el dueño (Oráculo) puede reportar
            assert_eq!(self.env().caller(), self.owner);

            // LÓGICA PHYFI:
            // Si el estrés físico supera el límite, pagamos automáticamente.
            if current_stress > self.stress_threshold && self.is_active {
                // Transferir todo el dinero al cliente
                self.env().transfer(self.insured, self.payout_amount).unwrap();
                // Cerrar el contrato
                self.is_active = false;
            }
        }
        
        /// Ver estado del seguro
        #[ink(message)]
        pub fn get_status(&self) -> bool {
            self.is_active
        }
    }
}