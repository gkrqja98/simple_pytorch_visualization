import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Spinner, Alert } from 'react-bootstrap';
import ModelArchitecture from './components/ModelArchitecture';
import IterationView from './components/IterationView';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [iterations, setIterations] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // 모델 정보 가져오기
        const modelResponse = await axios.get('/api/model_info');
        setModelData(modelResponse.data);

        // 시각화 데이터 가져오기
        const visualizationResponse = await axios.post('/api/run_visualization', { epochs: 3 });
        setIterations(visualizationResponse.data.iterations);
        
        setLoading(false);
      } catch (err) {
        setError('데이터를 가져오는 중 오류가 발생했습니다. 백엔드 서버가 실행 중인지 확인해주세요.');
        setLoading(false);
        console.error('Error fetching data:', err);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Container className="text-center py-5">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">로딩 중...</span>
        </Spinner>
        <p className="mt-3">CNN 모델 분석 중... 잠시만 기다려주세요.</p>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="py-5">
        <Alert variant="danger">
          <Alert.Heading>오류 발생!</Alert.Heading>
          <p>{error}</p>
        </Alert>
      </Container>
    );
  }

  return (
    <Container fluid>
      <header className="my-4">
        <h1 className="text-center">PyTorch CNN Visualization Tool</h1>
        <p className="text-center text-muted">
          Visualizing Forward and Backward Propagation of Conv2d, ReLU, MaxPool2d, Linear Layers
        </p>
      </header>

      <section className="model-section mb-5">
        <h2 className="section-title">1. Model Architecture</h2>
        {modelData && <ModelArchitecture modelData={modelData} />}
      </section>

      {iterations.map((iteration, index) => (
        <section key={index} className="iteration-section mb-5">
          <h2 className="section-title">Iteration {index + 1}</h2>
          <IterationView 
            iteration={iteration}
            iterationIndex={index}
          />
        </section>
      ))}

      <footer className="text-center py-4 mt-5 border-top">
        <p className="text-muted">
          PyTorch CNN Visualization Tool &copy; {new Date().getFullYear()}
        </p>
      </footer>
    </Container>
  );
}

export default App;
