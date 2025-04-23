## Testing the Model

There are two ways to test the model on your images:

### Option 1: Using the Interactive Menu

1. Run the main script:

```bash
python run.py
```

2. From the menu, select option `2` (Test model).

3. You'll be prompted to enter:
   - The test directory path (default: `./test`)
   - The model file path (default: `trained_cnn.pth`)

4. The script will test the model and display the classification accuracy for each category.

### Option 2: Direct Script Execution

You can also run the testing function directly by executing:

```bash
python scene_recog_cnn.py --phase test --test_dir ./test --model_path trained_cnn.pth
```

## Test Data Structure

Your test data should be organized as follows:
```
test/
  ├── bedroom/
  ├── Coast/
  ├── Forest/
  ├── Highway/
  ├── industrial/
  ├── Insidecity/
  ├── kitchen/
  ├── livingroom/
  ├── Mountain/
  ├── Office/
  ├── OpenCountry/
  ├── store/
  ├── Street/
  ├── Suburb/
  └── TallBuilding/
```

Each category folder should contain the test images for that class.


## Understanding Test Results

After testing, the script will display:
- Overall accuracy across all categories
- Per-class accuracy for each of the 15 scene categories

The results help you understand how well the model recognizes different types of scenes.