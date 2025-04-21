/**
 * 행렬 곱셈 함수
 * @param {Array} a - 첫 번째 행렬
 * @param {Array} b - 두 번째 행렬
 * @returns {Array} 곱셈 결과 행렬
 */
export const matrixMultiply = (a, b) => {
  // 행렬 차원 검증
  const aRows = a.length;
  const aCols = a[0].length;
  const bRows = b.length;
  const bCols = b[0].length;
  
  if (aCols !== bRows) {
    throw new Error(`행렬 곱셈 불가: 첫 번째 행렬의 열 수(${aCols})와 두 번째 행렬의 행 수(${bRows})가 일치하지 않습니다.`);
  }
  
  // 결과 행렬 초기화
  const result = Array(aRows).fill().map(() => Array(bCols).fill(0));
  
  // 행렬 곱셈 계산
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      for (let k = 0; k < aCols; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  
  return result;
};

/**
 * 행렬 전치 함수
 * @param {Array} matrix - 입력 행렬
 * @returns {Array} 전치된 행렬
 */
export const transpose = (matrix) => {
  const rows = matrix.length;
  const cols = matrix[0].length;
  
  // 결과 행렬 초기화
  const result = Array(cols).fill().map(() => Array(rows).fill(0));
  
  // 전치 계산
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  
  return result;
};

/**
 * 합성곱 연산 구현
 * @param {Array} input - 입력 텐서 (2D 배열)
 * @param {Array} kernel - 커널 (2D 배열)
 * @returns {Array} 합성곱 결과
 */
export const convolve2d = (input, kernel) => {
  const inputHeight = input.length;
  const inputWidth = input[0].length;
  const kernelHeight = kernel.length;
  const kernelWidth = kernel[0].length;
  
  // 출력 크기 계산
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  
  // 결과 배열 초기화
  const output = Array(outputHeight).fill().map(() => Array(outputWidth).fill(0));
  
  // 합성곱 연산 수행
  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      for (let ki = 0; ki < kernelHeight; ki++) {
        for (let kj = 0; kj < kernelWidth; kj++) {
          output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
        }
      }
    }
  }
  
  return output;
};

/**
 * ReLU 활성화 함수
 * @param {Array} x - 입력 배열 (다차원 배열 가능)
 * @returns {Array} ReLU 적용 결과
 */
export const relu = (x) => {
  // 1D 배열인 경우
  if (!Array.isArray(x[0])) {
    return x.map(val => Math.max(0, val));
  }
  
  // 2D 배열인 경우
  return x.map(row => row.map(val => Math.max(0, val)));
};

/**
 * 최대 풀링 연산
 * @param {Array} input - 입력 텐서 (2D 배열)
 * @param {number} poolSize - 풀링 크기
 * @param {number} stride - 스트라이드
 * @returns {Object} 풀링 결과 및 인덱스
 */
export const maxPool2d = (input, poolSize = 2, stride = 2) => {
  const height = input.length;
  const width = input[0].length;
  
  // 출력 크기 계산
  const outputHeight = Math.floor((height - poolSize) / stride) + 1;
  const outputWidth = Math.floor((width - poolSize) / stride) + 1;
  
  // 결과 및 인덱스 배열 초기화
  const output = Array(outputHeight).fill().map(() => Array(outputWidth).fill(0));
  const indices = Array(outputHeight).fill().map(() => Array(outputWidth).fill(0));
  
  // 최대 풀링 연산 수행
  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      let maxVal = -Infinity;
      let maxIdx = -1;
      
      for (let ki = 0; ki < poolSize; ki++) {
        for (let kj = 0; kj < poolSize; kj++) {
          const ii = i * stride + ki;
          const jj = j * stride + kj;
          
          if (ii < height && jj < width) {
            if (input[ii][jj] > maxVal) {
              maxVal = input[ii][jj];
              maxIdx = ii * width + jj; // 평탄화된 인덱스
            }
          }
        }
      }
      
      output[i][j] = maxVal;
      indices[i][j] = maxIdx;
    }
  }
  
  return { output, indices };
};

/**
 * 소프트맥스 함수
 * @param {Array} x - 입력 배열
 * @returns {Array} 소프트맥스 확률 분포
 */
export const softmax = (x) => {
  const expValues = x.map(val => Math.exp(val));
  const sumExp = expValues.reduce((acc, val) => acc + val, 0);
  return expValues.map(val => val / sumExp);
};

/**
 * 교차 엔트로피 손실 계산
 * @param {Array} predictions - 모델 예측값 (소프트맥스 출력)
 * @param {number} target - 타겟 클래스 인덱스
 * @returns {number} 손실값
 */
export const crossEntropyLoss = (predictions, target) => {
  // 타겟 클래스의 예측 확률값
  const targetProb = predictions[target];
  
  // 손실값 계산: -log(targetProb)
  return -Math.log(targetProb);
};
