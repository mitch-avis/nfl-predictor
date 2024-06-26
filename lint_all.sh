#!/bin/bash

echo "Sorting imports with isort..."
isort nfl_predictor/

echo "Reformatting with black..."
black nfl_predictor/

echo "Linting with flake8..."
flake8 nfl_predictor/

echo "Linting with pylint..."
pylint nfl_predictor/
