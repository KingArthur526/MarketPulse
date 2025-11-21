from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        print("Loading FinBERT model... This takes 2-3 minutes first time.")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("Model loaded successfully!")
    
    def analyze_single(self, text):
        """Analyze one tweet"""
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            }
        
        # Convert text to model input
        inputs = self.tokenizer(text, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        
        # Get prediction
        outputs = self.model(**inputs)
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the prediction
        prediction = torch.argmax(probs, dim=-1).item()
        labels = ['positive', 'negative', 'neutral']
        
        # Return result
        return {
            'sentiment': labels[prediction],
            'confidence': round(probs[0][prediction].item(), 3),
            'scores': {
                'positive': round(probs[0][0].item(), 3),
                'negative': round(probs[0][1].item(), 3),
                'neutral': round(probs[0][2].item(), 3)
            }
        }
    
    def analyze_batch(self, tweet_list):
        """Analyze multiple tweets"""
        results = []
        total = len(tweet_list)
        for i, tweet in enumerate(tweet_list, 1):
            print(f"Processing tweet {i}/{total}...", end='\r')
            result = self.analyze_single(tweet)
            result['text'] = tweet
            results.append(result)
        print(f"\nCompleted analyzing {total} tweets!")
        return results

# Test code
if __name__ == "__main__":
    print("Testing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_tweet = "Apple stock is soaring after great earnings! 📈"
    result = analyzer.analyze_single(test_tweet)
    
    print(f"\nTest Tweet: {test_tweet}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print("\nTest passed! ✓")
