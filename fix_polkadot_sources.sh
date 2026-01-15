#!/usr/bin/env bash
set -euo pipefail

echo "== (1) Eliminar cualquier [patch] que apunte a polkadot-sdk con branch release-polkadot-v1.1.0 =="
perl -0777 -i -pe 's/\n\[patch\."https:\/\/github\.com\/paritytech\/polkadot-sdk\.git\?branch=release-polkadot-v1\.1\.0"\][^\[]*//sg' Cargo.toml

echo "== (2) Reemplazar en TODO el workspace cualquier uso de branch=release-polkadot-v1.1.0 por tag=polkadot-v1.6.0 =="
find . -name Cargo.toml -print0 | xargs -0 perl -pi -e 's/\bbranch\s*=\s*"release-polkadot-v1\.1\.0"\s*,\s*tag\s*=\s*"[^"]*"\s*/tag = "polkadot-v1.6.0" /g'
find . -name Cargo.toml -print0 | xargs -0 perl -pi -e 's/\btag\s*=\s*"[^"]*"\s*,\s*branch\s*=\s*"release-polkadot-v1\.1\.0"\s*/tag = "polkadot-v1.6.0" /g'
find . -name Cargo.toml -print0 | xargs -0 perl -pi -e 's/\bbranch\s*=\s*"release-polkadot-v1\.1\.0"\s*/tag = "polkadot-v1.6.0"/g'

echo "== (3) Verificación: NO debe quedar release-polkadot-v1.1.0 en ningún Cargo.toml =="
grep -Rni 'release-polkadot-v1\.1\.0' . --include Cargo.toml || true

echo "== (4) Limpieza total de cache + lockfile =="
rm -rf ~/.cargo/git ~/.cargo/registry
rm -f Cargo.lock
cargo clean

echo "== (5) Regenerar lockfile =="
cargo generate-lockfile

echo "== (6) Verificar que Cargo.lock NO contiene branch=release-polkadot-v1.1.0 =="
grep -n 'branch=release-polkadot-v1\.1\.0' Cargo.lock || true

echo "== (7) Mostrar las entradas de polkadot-sdk en el lock (debería ser solo polkadot-v1.6.0) =="
grep -n 'paritytech/polkadot-sdk\.git' Cargo.lock | sed -n '1,160p'

echo "== (8) Ver qué features están activando sc-network (clave para el E0080) =="
cargo tree -i sc-network -e features | sed -n '1,220p'

echo "== (9) Build =="
cargo build -p trionchain-node --release
