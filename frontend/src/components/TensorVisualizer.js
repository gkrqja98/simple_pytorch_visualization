import React from 'react';
import { OverlayTrigger, Tooltip } from 'react-bootstrap';

const TensorVisualizer = ({ tensor }) => {
  // 텐서가 없거나 유효하지 않은 경우 처리
  if (!tensor || !Array.isArray(tensor)) {
    return <div>유효한 텐서 데이터가 없습니다.</div>;
  }
  
  // 값의 포맷팅 함수 - 소수점 자릿수 조절 및 표시 최적화
  const formatValue = (value) => {
    // 전체 값은 툴팁에 표시할 정확한 값
    const fullPrecision = value.toFixed(6);
    
    // 화면에 표시될 값 - 크기에 따라 다르게 처리
    let displayValue;
    
    if (Math.abs(value) < 0.0001) {
      // 매우 작은 값은 과학적 표기법 사용
      displayValue = value.toExponential(2);
    } else if (Math.abs(value) < 0.01) {
      // 작은 값은 최대 4자리까지 표시
      displayValue = value.toFixed(4);
    } else {
      // 일반적인 값은 2자리까지 표시
      displayValue = value.toFixed(2);
    }
    
    return { displayValue, fullPrecision };
  };
  
  // 1D 텐서 처리
  if (!Array.isArray(tensor[0])) {
    return (
      <div className="tensor-visualization">
        <div className="tensor-row">
          {tensor.map((value, i) => {
            const { displayValue, fullPrecision } = formatValue(value);
            return (
              <OverlayTrigger
                key={i}
                placement="top"
                overlay={<Tooltip>{fullPrecision}</Tooltip>}
              >
                <div className="tensor-cell">{displayValue}</div>
              </OverlayTrigger>
            );
          })}
        </div>
      </div>
    );
  }
  
  // 2D 텐서 시각화
  return (
    <div className="tensor-visualization">
      {tensor.map((row, i) => (
        <div key={i} className="tensor-row">
          {row.map((value, j) => {
            const { displayValue, fullPrecision } = formatValue(value);
            return (
              <OverlayTrigger
                key={j}
                placement="top"
                overlay={<Tooltip>{fullPrecision}</Tooltip>}
              >
                <div 
                  className="tensor-cell"
                  style={{
                    backgroundColor: `rgba(0, 123, 255, ${Math.abs(value) / 10})`,
                    color: Math.abs(value) > 5 ? 'white' : 'black'
                  }}
                >
                  {displayValue}
                </div>
              </OverlayTrigger>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export default TensorVisualizer;
