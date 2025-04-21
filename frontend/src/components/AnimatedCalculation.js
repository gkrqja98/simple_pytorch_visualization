import React, { useState, useEffect } from 'react';
import { InlineMath } from 'react-katex';
import { Button } from 'react-bootstrap';

/**
 * AnimatedCalculation component for step-by-step visualization of calculations
 * @param {Object} props Component properties
 * @param {Array} props.steps Array of calculation steps, each with equation and result
 * @param {number} props.interval Time interval between steps in ms
 */
const AnimatedCalculation = ({ steps, interval = 1500 }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          // Reset to first step after completing all steps
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, interval);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, steps.length, interval]);
  
  // Animated highlighting for the current step
  useEffect(() => {
    // Only animate highlighting when a step changes
    if (currentStep > 0) {
      const step = steps[currentStep];
      if (step.highlights) {
        // Cycle through the highlights
        let highlightTimer;
        let idx = 0;
        
        const animateHighlights = () => {
          setHighlightIndex(step.highlights[idx]);
          idx = (idx + 1) % step.highlights.length;
          highlightTimer = setTimeout(animateHighlights, 500); // Highlight each term for 500ms
        };
        
        animateHighlights();
        return () => {
          clearTimeout(highlightTimer);
          setHighlightIndex(-1);
        };
      }
    }
  }, [currentStep, steps]);
  
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };
  
  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setCurrentStep(0); // Loop back to start
    }
  };
  
  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    } else {
      setCurrentStep(steps.length - 1); // Loop to end
    }
  };
  
  // Render no steps available message if no steps provided
  if (!steps || steps.length === 0) {
    return <div>No calculation steps available.</div>;
  }
  
  const currentStepData = steps[currentStep];
  
  return (
    <div className="animated-calculation">
      <div className="calculation-display p-3 mb-3 bg-light rounded">
        {currentStepData.description && (
          <p className="mb-2">{currentStepData.description}</p>
        )}
        
        <div className="equation-display my-3">
          <InlineMath math={currentStepData.equation} />
        </div>
        
        {currentStepData.result && (
          <div className="result-display mt-2">
            <strong>Result:</strong> {currentStepData.result}
          </div>
        )}
      </div>
      
      <div className="controls d-flex justify-content-between">
        <Button 
          variant="outline-secondary" 
          size="sm" 
          onClick={handlePrevStep}
        >
          &laquo; Previous
        </Button>
        
        <Button 
          variant={isPlaying ? "outline-danger" : "outline-primary"} 
          size="sm" 
          onClick={handlePlayPause}
        >
          {isPlaying ? "Pause" : "Play Animation"}
        </Button>
        
        <Button 
          variant="outline-secondary" 
          size="sm" 
          onClick={handleNextStep}
        >
          Next &raquo;
        </Button>
      </div>
      
      <div className="step-indicator text-center mt-2 text-muted">
        Step {currentStep + 1} of {steps.length}
      </div>
    </div>
  );
};

export default AnimatedCalculation;
