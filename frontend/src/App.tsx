import { useState } from 'react';

// Define a type for our API response for type safety
interface PredictionResult {
  prediction: string;
  confidence: string;
}

function App() {
  const [text, setText] = useState<string>('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleClassify = async () => {
    if (!text.trim()) {
      alert("Please enter some text to classify.");
      return;
    }
    
    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);

    } catch (error) {
      console.error("Error fetching data: ", error);
      setResult({ prediction: 'API Error', confidence: 'Could not connect to the model API.' });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Dynamically determine result box styling
  const resultBoxStyle = result
    ? result.prediction === 'Toxic'
      ? 'bg-red-900/50 border-red-500'
      : 'bg-green-900/50 border-green-500'
    : '';


  return (
    <main className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl text-center">
        
        <h1 className="text-4xl md:text-5xl font-bold text-cyan-300">
          Toxic Comment Classifier
        </h1>
        <p className="mt-4 text-lg text-gray-300">
          Powered by a fine-tuned base BERT model.
        </p>

        <p className="mt-4 text-sm text-gray-400 max-w-xl mx-auto">
          This model was trained on the{' '}
          <a 
            href="https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification"
            target="_blank" 
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            Jigsaw Unintended Bias in Toxicity Classification
          </a>
          {' '}dataset from Kaggle. The training data was labeled by paid annotators considering{' '}
          <span className="text-red-400">
            toxic 
          </span>
          {' '}as{' '} 
          <span className="text-gray-200">
            "anything rude, disrespectful or otherwise likely to make someone leave a discussion."
          </span>
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste your message here..."
          className="mt-8 w-full h-40 p-4 bg-gray-800 border-2 border-gray-600 rounded-lg focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400 transition-colors"
        />

        <button 
          onClick={handleClassify} 
          disabled={isLoading}
          className="mt-4 w-full py-3 px-6 bg-cyan-500 text-gray-900 font-bold rounded-lg text-xl hover:bg-cyan-400 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Classifying...' : 'Classify Text'}
        </button>

        {result && (
          <div className={`mt-8 p-6 rounded-lg border-2 text-left ${resultBoxStyle}`}>
            <h2 className="text-2xl font-semibold">
              Result: <span className={result.prediction === 'Toxic' ? 'text-red-400' : 'text-green-400'}>{result.prediction}</span>
            </h2>
            <p className="mt-2 text-gray-300 text-lg">
              Confidence: {result.confidence}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}

export default App;