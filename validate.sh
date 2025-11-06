python validation/make_input.py
mv validation/input.json input.json

python solution.py

mv output.json validation/output.json
python validation/score_submission.py

rm input.json
rm validation/output.json