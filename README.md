## Project Aim  

This project aims to catch **keyword spamming 🍖** — when sellers list large numbers of unrelated or irrelevant keywords in an item’s description to boost its ranking in search results 📈.  

For example, a buyer searching for “Levi jeans” 👖 might see Diesel jeans ⛽️ ranked highly, simply because the word “Levi” was spammed in the description. This is frustrating for the buyer 😖.  

Here are examples of product descriptions with keyword spam:  

```
Low waist/rise diesel bootcut/flared jeans. Size XS/6. Great condition. Cool red stitching details.
Message for any questions :) UK shipping only
No returns
#vintage #diesel #denim #lowrise #levi #wrangler #lee #y2k #90s #2010s #blue #black #faded
```

```
Low rise y2k blue Diesel bootcut jeans  
Size label W29 L32  
Flat laid measurements below —  
32 inch waist (sits on hips)  
7 inch rise  
32 inch inseam  
FREE UK SHIP  
£15 international  
Ignore: 80s 90s y2k baggy navy jeans denim levi calvin klein
```

If we can classify item descriptions as **‘spammy’ 🍖**, we can **demote those items in the ranking algorithm 📉**.  
This project is focused on building that **classifier 🔨**.

---

## Summary and Next Steps

### What We Accomplished:

1. **Fixed Critical Data Leakage**: Proper train/test separation prevents invalid performance metrics  
2. **Implemented Proper ML Pipeline**: Feature engineering → training → validation → inference  
3. **Enhanced Feature Engineering**: Comprehensive features without losing important information  
4. **Improved Model Architecture**: Properly tuned XGBoost with realistic hyperparameters  
5. **Added Interpretability**: Feature importance analysis helps understand spam patterns  
6. **Created Production-Ready Code**: Modular, maintainable, and extensible  
7. **Explored Image–Text Consistency**: Added potential to compare product images with descriptions using vision-language models such as [Qwen3-VL (GitHub link)](https://github.com/QwenLM/Qwen3-VL)  

### Training Process Summary:
1. **Data Loading**: Separate train/test datasets  
2. **Feature Extraction**: Basic stats + TF-IDF + named entities  
3. **Feature Scaling**: Normalize features for model training  
4. **Model Training**: XGBoost with proper hyperparameters  
5. **Validation**: Cross-validation and realistic metrics  

### Inference Process Summary:
1. **New Data Processing**: Clean and normalize input descriptions  
2. **Feature Extraction**: Use fitted transformers (no retraining)  
3. **Feature Scaling**: Apply fitted scaler  
4. **Prediction**: Get both labels and probabilities  
5. **Interpretation**: Analyze feature contributions  
6. **Optional Image–Text Check**: Compare product photos with their descriptions using vision-language models like [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)  

### Next Steps for Production:

**Short-term (1-2 weeks):**  
- Hyperparameter tuning with Optuna  
- A/B testing framework  
- Model monitoring setup  
- Performance optimization  

**Medium-term (1-2 months):**  
- Deep learning approaches (BERT/RoBERTa)  
- Active learning for data collection  
- Multi-modal features (images + text)  
- Real-time inference optimization  
- **Image–Text Consistency Checks**: Integrate models like [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) to detect mismatches between descriptions and product images  

**Long-term (3+ months):**  
- Automated spam remediation workflows  
- Business intelligence dashboards  
- Continuous learning system  
- Advanced explainability features  

### Key Insight:
**Proper ML practices > High accuracy metrics**

The original 95% accuracy was meaningless due to data leakage. Our realistic 70-80% accuracy will actually work in production and help solve the real business problem of keyword spam detection — and by adding **image–text comparison**, we can further improve fraud and spam detection by flagging products whose descriptions don’t match their photos.

> **🔗 Qwen3-VL GitHub Repository:** [https://github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

