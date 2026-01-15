use crate::{
	chain_spec,
	cli::{Cli, Subcommand},
	service,
};

use sc_cli::{ChainSpec, SubstrateCli};

impl SubstrateCli for Cli {
	fn impl_name() -> String { "TrionChain Node".into() }
	fn impl_version() -> String { env!("SUBSTRATE_CLI_IMPL_VERSION").into() }
	fn description() -> String { env!("CARGO_PKG_DESCRIPTION").into() }
	fn author() -> String { env!("CARGO_PKG_AUTHORS").into() }
	fn support_url() -> String { "https://github.com/".into() }
	fn copyright_start_year() -> i32 { 2025 }

	fn load_spec(&self, id: &str) -> Result<Box<dyn ChainSpec>, String> {
		Ok(match id {
			"dev" => Box::new(chain_spec::development_config()?),
			"local" => Box::new(chain_spec::local_testnet_config()?),
			"" => Box::new(chain_spec::development_config()?),
			path => Box::new(chain_spec::ChainSpec::from_json_file(std::path::PathBuf::from(path))?),
		})
	}}

pub fn run() -> sc_cli::Result<()> {
	let cli = Cli::from_args();

	match &cli.subcommand {
		Some(Subcommand::Key(cmd)) => cmd.run(&cli),
		Some(_) => Err(sc_cli::Error::Input("Subcommand disabled temporarily (focus: --dev node build)".into())),
		None => {
			let runner = cli.create_runner(&cli.run)?;
			runner.run_node_until_exit(|config| async move {
				service::new_full(config).map_err(sc_cli::Error::Service)
			})
		}
	}
}
