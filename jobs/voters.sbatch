#!/usr/bin/env bash
#SBATCH --job-name=voters
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "$HOME/Git/Github/consensus"
cargo fmt --all --check && echo FMT-CLEAN
cargo clippy --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)" -A8 | head -40
cargo build --release --example synthetic_voters 2>&1 | grep -E "^error" -A10 | head -30
for spec in "9 400 20" "5 400 20" "15 400 20" "9 60 20"; do
  echo "=== $spec"
  ./target/release/examples/synthetic_voters $spec
done
echo ONE-DONE
