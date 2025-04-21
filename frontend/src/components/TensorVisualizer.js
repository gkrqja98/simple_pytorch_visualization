import React from 'react';

const TensorVisualizer = ({ tensor }) => {
  // 텐서가 없거나 유효하지 않은 경우 처리
  if (!tensor || !Array.isArray(tensor)) {
    return <div>유효한 텐서 데이터가 없습니다.</div>;
  }
  
  // 1D 텐서 처리
  if (!Array.isArray(tensor[0])) {
    return (
      <div className="tensor-visualization">
        <div className="tensor-row">
          {tensor.map((value, i) => (
            <div key={i} className="tensor-cell">{value.toFixed(2)}</div>
          ))}
        </div>
      </div>
    );
  }
  
  // 2D 텐서 시각화
  return (
    <div className="tensor-visualization">
      {tensor.map((row, i) => (
        <div key={i} className="tensor-row">
          {row.map((value, j) => (
            <div 
              key={j} 
              className="tensor-cell"
              style={{
                backgroundColor: `rgba(0, 123, 255, ${Math.abs(value) / 10})`,
                color: Math.abs(value) > 5 ? 'white' : 'black'
              }}
            >
              {value.toFixed(2)}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default TensorVisualizer;
