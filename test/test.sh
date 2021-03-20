cd ..
source env.sh
python -m causal_discovery \
    --header=0 \
    --output-dir=test/test_data \
    test/test_data/simul_data.csv \
    3 \
    matrixT