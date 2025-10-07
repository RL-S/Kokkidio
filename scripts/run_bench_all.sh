#!/bin/bash

set -eu

# script path
sd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

isQuickTest=${1:-false}
quickTestFlag=""
if [[ $isQuickTest == true ]]; then
	quickTestFlag="--quick"
fi

backend=${2:-def}

runscr="${sd}/run_bench.sh"

if [[ $backend == "def" ]]; then
	# the node file should define a default backend,
	# so let's retrieve it
	env_dir="${sd}/../env"
	node_name=$(uname -n)
	node_pat="$env_dir/node_patterns.sh"
	if [ -f "$node_pat" ]; then
		source "$node_pat"
	else
		echo "Node pattern file not found."
	fi

	nodefile="$env_dir/nodes/${node_name}.sh"
	echo "Node file: $nodefile"
	if [ -f "$nodefile" ]; then
		source "$nodefile"
		backends=($backend_default)
	else
		echo "Node file not found."
	fi
elif [[ $backend == "all" ]]; then
	backends=(cuda hip ompt sycl)
else
	backends=($backend)
fi

run_all () {
	local example=$1
	for target in gpu cpu; do
# 	for target in cpu; do
		if [[ "$target" == "gpu" ]] || [[ $isQuickTest == true ]]; then
			np=(4)
		else
			np=(1 2 4)
# 			np=(8 12 16 24 32 40 48 64)
		fi
		for p in "${np[@]}"; do
			for scalar in float double; do
# 			for scalar in float; do
				for b in "${backends[@]}"; do
					"$runscr" \
						--backend $b \
						--scalar $scalar \
						--example $example \
						--target $target \
						--nCores $p \
						$quickTestFlag
				done
			done
		done
	done
}

run_all axpy
run_all dotProduct
run_all norm
run_all friction
run_all rpow
run_all raxpy
