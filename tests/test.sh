cd ..
source env.sh

if [ ! -d "tests/data" ]; then
  mkdir tests/data
fi

python -m causal_discovery fast-simul-data --output-dir tests/data
python -m causal_discovery run-local-ng-cd \
    --header=0 \
    --output-dir=tests/data \
    tests/data/simul_data.csv \
    3 \
    matrixT