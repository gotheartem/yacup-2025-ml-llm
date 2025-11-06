python validation/test_input.py
mv validation/input.json input.json

python solution.py

mv output.json validation/output.json
python validation/test_checker.py

rm input.json
rm validation/output.json