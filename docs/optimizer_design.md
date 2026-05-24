# DSPy Optimizer Tasarımı: DRG için İteratif Öğrenme

## 1. Genel Bakış

DSPy optimizer entegrasyonu, DRG extraction pipeline'ının iterative learning ile sürekli iyileştirilmesini sağlar. Bu modül, DSPy'nin güçlü optimizer'larını kullanarak extraction kalitesini artırır.

## 2. Optimizer Seçimi

### 2.1 BootstrapFewShot (Varsayılan)

**Algoritma:**
1. Training examples'dan bootstrapped demonstrations oluştur
2. Her example için en iyi few-shot examples'ı seç
3. LLM'e bu examples'ları göster
4. Performance'a göre examples'ları optimize et

**Avantajlar:**
- Hızlı (az sayıda training example yeterli)
- Labeled data gerektirmez (self-bootstrapping)
- Genel amaçlı, çoğu task için çalışır

**Dezavantajlar:**
- Quality training examples'a bağımlı
- Complex patterns için yetersiz olabilir

**Kullanım Senaryoları:**
- Limited training data
- Quick prototyping
- General-purpose extraction tasks

### 2.2 MIPRO (Multi-prompt Instruction Proposal and Refinement Optimization)

**Algoritma:**
1. Multiple prompt candidates oluştur
2. Her candidate'ı evaluate et
3. Best candidates'ı refine et
4. Iterative olarak improve et

**Avantajlar:**
- Prompt optimization
- High-quality results
- Systematic exploration

**Dezavantajlar:**
- Yavaş (multiple candidates evaluate etmek gerekir)
- Computational cost yüksek

**Kullanım Senaryoları:**
- Quality-critical applications
- Complex extraction tasks
- When prompt engineering is important

### 2.3 COPRO (Compositional Prompt Optimization)

**Algoritma:**
1. Prompt'u compositional parts'a böl
2. Her part'ı optimize et
3. Parts'ları birleştir
4. End-to-end optimize et

**Avantajlar:**
- Modular optimization
- Better interpretability
- Systematic approach

**Dezavantajlar:**
- Complex implementation
- Yavaş

**Kullanım Senaryoları:**
- Complex multi-step extraction
- When modularity is important

### 2.4 LabeledFewShot

**Algoritma:**
1. Labeled training examples kullan
2. Fixed number of examples seç
3. LLM'e göster

**Avantajlar:**
- Basit
- Hızlı
- Deterministic

**Dezavantajlar:**
- Labeled data gerektirir
- Adaptive değil

**Kullanım Senaryoları:**
- Abundant labeled data
- Simple extraction tasks
- When determinism is important

## 3. Evaluation Metrics

### 3.1 Entity Extraction Metrics

**Precision:**
```
entity_precision = |expected_entities ∩ predicted_entities| / |predicted_entities|
```

**Recall:**
```
entity_recall = |expected_entities ∩ predicted_entities| / |expected_entities|
```

**F1 Score:**
```
entity_f1 = 2 * (precision * recall) / (precision + recall)
```

### 3.2 Relation Extraction Metrics

**Precision:**
```
relation_precision = |expected_relations ∩ predicted_relations| / |predicted_relations|
```

**Recall:**
```
relation_recall = |expected_relations ∩ predicted_relations| / |expected_relations|
```

**F1 Score:**
```
relation_f1 = 2 * (precision * recall) / (precision + recall)
```

### 3.3 Combined Metrics

**Weighted Average:**
```
precision = 0.6 * entity_precision + 0.4 * relation_precision
recall = 0.6 * entity_recall + 0.4 * relation_recall
f1 = 0.6 * entity_f1 + 0.4 * relation_f1
```

**Rationale:**
- Entity extraction genellikle relation extraction'dan daha kolay
- Entity'ler relation'ların temelidir
- 60/40 weighting entity'leri biraz daha önemli kılar

**Accuracy:**
```
accuracy = (correct_entities + correct_relations) / (total_expected_entities + total_expected_relations)
```

## 4. Iterative Improvement Loop

### 4.1 Loop Structure

```
1. Initialize optimizer with training examples
2. For each iteration:
   a. Optimize extractor using current training set
   b. Evaluate on test set
   c. Check if threshold met
   d. If not, continue to next iteration
3. Return best extractor
```

### 4.2 Stopping Criteria

**Threshold-based:**
- F1 score >= target threshold (default: 0.7)
- Early stopping if threshold met

**Iteration-based:**
- Maximum iterations (default: 5)
- Stop after max iterations

**Convergence-based:**
- Stop if improvement < epsilon between iterations
- Prevents overfitting

### 4.3 Training Set Expansion

**Active Learning:**
- Her iteration'da yeni examples ekle
- Model'in en çok hata yaptığı examples'ları seç
- Human-in-the-loop için hazır

**Self-Bootstrapping:**
- Model'in kendi predictions'larını training'e ekle
- High-confidence predictions'ları kullan
- Iterative refinement

## 5. Before/After Comparison

### 5.1 Comparison Metrics

**Absolute Improvement:**
```
improvement = after_metric - before_metric
```

**Percentage Improvement:**
```
improvement_percent = (after_metric - before_metric) / before_metric * 100
```

### 5.2 Comparison Report

**Structure:**
- Base metrics (before optimization)
- Optimized metrics (after optimization)
- Absolute improvements
- Percentage improvements
- Per-category breakdown (entity vs relation)

## 6. DSPy Signature Integration

### 6.1 Dynamic Signature Creation

DRG optimizer, mevcut `KGExtractor`'ın signature'larını kullanır:

**Entity Extraction Signature:**
```
EntityExtraction:
  text: str = InputField
  entities: List[Tuple[str, str]] = OutputField
```

**Relation Extraction Signature:**
```
RelationExtraction:
  text: str = InputField
  entities: List[Tuple[str, str]] = InputField
  relations: List[Tuple[str, str, str]] = OutputField
```

### 6.2 Optimizer Compatibility

**BootstrapFewShot:**
- Works with any DSPy Module
- Automatically selects best demonstrations
- Compatible with ChainOfThought

**MIPRO/COPRO:**
- Requires Module with forward() method
- Can optimize prompt templates
- Works with KGExtractor

## 7. Configuration Management

### 7.1 OptimizerConfig

**Parameters:**
- `optimizer_type`: Optimizer selection
- `max_bootstrapped_demos`: Number of bootstrapped examples
- `max_labeled_demos`: Number of labeled examples
- `num_candidates`: Number of candidates (MIPRO/COPRO)
- `init_temperature`: Initial temperature for exploration
- `metric_threshold`: Target metric value
- `max_iterations`: Maximum optimization iterations

### 7.2 Default Configuration

**BootstrapFewShot:**
- max_bootstrapped_demos: 4
- max_labeled_demos: 16

**MIPRO/COPRO:**
- num_candidates: 10
- init_temperature: 1.0

**General:**
- metric_threshold: 0.7
- max_iterations: 5

## 8. Training Example Format

### 8.1 Example Structure

```python
{
    "text": "Input text to extract from",
    "expected_entities": [
        ("entity_name", "entity_type"),
        ...
    ],
    "expected_relations": [
        ("source", "relation", "target"),
        ...
    ]
}
```

### 8.2 Example Quality Guidelines

**Good Examples:**
- Clear entity boundaries
- Explicit relations
- Representative of target domain
- Diverse patterns

**Bad Examples:**
- Ambiguous entities
- Implicit relations
- Outlier cases
- Too similar to each other

## 9. Evaluation Protocol

### 9.1 Train/Test Split

**Recommended Split:**
- Training: 70-80% of examples
- Test: 20-30% of examples
- Validation: Optional, 10% of training

**Cross-Validation:**
- K-fold cross-validation for small datasets
- Stratified sampling for balanced evaluation

### 9.2 Evaluation Frequency

**Per Iteration:**
- Evaluate after each optimization step
- Track metrics over time
- Identify overfitting

**Final Evaluation:**
- Evaluate on held-out test set
- Compare all iterations
- Select best model

## 10. Trade-offs

### 10.1 Optimizer Selection

| Optimizer | Speed | Quality | Data Requirements | Use Case |
|-----------|-------|---------|-------------------|----------|
| BootstrapFewShot | Fast | Good | Low | General purpose |
| MIPRO | Slow | Excellent | Medium | Quality-critical |
| COPRO | Slow | Excellent | Medium | Complex tasks |
| LabeledFewShot | Fast | Good | High | Simple tasks |

### 10.2 Training Set Size

**Small (< 10 examples):**
- BootstrapFewShot recommended
- Quick iteration
- Limited improvement potential

**Medium (10-50 examples):**
- BootstrapFewShot or MIPRO
- Good improvement potential
- Balanced speed/quality

**Large (> 50 examples):**
- MIPRO or COPRO
- Maximum improvement potential
- Can afford slower optimization

### 10.3 Iteration Count

**Few Iterations (1-3):**
- Fast
- Limited improvement
- Good for prototyping

**Many Iterations (5-10):**
- Slower
- Better improvement
- Risk of overfitting

## 11. Best Practices

### 11.1 Training Example Selection

1. **Diversity**: Cover different patterns and edge cases
2. **Quality**: Ensure accurate ground truth labels
3. **Representativeness**: Match target domain distribution
4. **Balance**: Equal representation of entity types and relations

### 11.2 Optimization Strategy

1. **Start Simple**: Begin with BootstrapFewShot
2. **Iterate**: Run multiple iterations
3. **Evaluate**: Monitor metrics on test set
4. **Refine**: Adjust configuration based on results
5. **Upgrade**: Switch to MIPRO/COPRO if needed

### 11.3 Evaluation Strategy

1. **Hold-out Test Set**: Never use test set for training
2. **Multiple Metrics**: Track precision, recall, F1, accuracy
3. **Per-category Analysis**: Entity vs relation performance
4. **Error Analysis**: Identify failure patterns
5. **Before/After Comparison**: Quantify improvements

## 12. Implementation Considerations

### 12.1 Performance

**Optimization Time:**
- BootstrapFewShot: ~1-5 minutes (depending on examples)
- MIPRO: ~10-30 minutes
- COPRO: ~15-45 minutes

**Inference Time:**
- Optimized extractor: Similar to base extractor
- No significant latency increase

### 12.2 Cost

**API Calls:**
- BootstrapFewShot: ~N * M calls (N=examples, M=demos)
- MIPRO: ~Candidates * Examples calls
- COPRO: ~Parts * Candidates * Examples calls

**Cost Optimization:**
- Cache demonstrations
- Limit candidate exploration
- Use smaller models for optimization

### 12.3 Scalability

**Training Set Size:**
- BootstrapFewShot: Scales well (O(N log N))
- MIPRO: Slower scaling (O(N²))
- COPRO: Slowest scaling (O(N³))

**Recommendation:**
- Start with small training set
- Expand iteratively
- Use active learning for large datasets

## 13. Future Directions

### 13.1 Advanced Optimizers

- **Bayesian Optimization**: Probabilistic approach
- **Reinforcement Learning**: RL-based prompt optimization
- **Neural Architecture Search**: Automatic architecture optimization

### 13.2 Multi-Task Learning

- Optimize for multiple extraction tasks simultaneously
- Shared representations
- Transfer learning

### 13.3 Online Learning

- Continuous learning from new examples
- Adaptive optimization
- Real-time improvement

