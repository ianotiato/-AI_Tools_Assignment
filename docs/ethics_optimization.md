# Part 3: Ethics & Optimization

## Ethical Considerations

### MNIST Model Biases:

1. **Cultural Bias**: Handwritten digits may vary across cultures
2. **Writing Style Bias**: Model might perform poorly on unusual handwriting styles
3. **Demographic Bias**: Training data may underrepresent certain demographic groups

### Amazon Reviews Model Biases:

1. **Language Bias**: Only analyzes English text
2. **Cultural Context**: Sentiment words may have different meanings across cultures
3. **Product Bias**: May favor popular brands with more training data

### Mitigation Strategies:

1. **TensorFlow Fairness Indicators**:

   - Analyze performance across different subgroups
   - Identify disparity in metrics
   - Set fairness constraints during training

2. **spaCy's Rule-based Systems**:
   - Add custom rules for domain-specific entities
   - Implement contextual sentiment analysis
   - Use diverse training data

## Troubleshooting Challenge

Common TensorFlow errors and solutions:

1. **Dimension Mismatches**:

   - Use `tf.reshape()` or `tf.expand_dims()`
   - Check input shapes with `model.summary()`

2. **Incorrect Loss Functions**:

   - Use `sparse_categorical_crossentropy` for integer labels
   - Use `categorical_crossentropy` for one-hot encoded labels

3. **Gradient Issues**:
   - Use gradient clipping: `tf.clip_by_value()`
   - Check for NaN values in inputs
