@echo off

set algorithm_id=0

python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 0
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 1
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 2
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 3
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 4
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 5
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 6
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 7
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 8
python -B train%algorithm_id%.py --base-dir .. --is-train 1 --split-id 9

pause
