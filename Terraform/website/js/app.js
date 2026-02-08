document.addEventListener('DOMContentLoaded', function() {
    // Элементы DOM
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const fileInfo = document.getElementById('fileInfo');
    const audioPreview = document.getElementById('audioPreview');
    const audioPlayer = document.getElementById('audioPlayer');
    const processBtn = document.getElementById('processBtn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    
    // Счетчики
    const selectedCount = document.getElementById('selectedCount');
    const processCount = document.getElementById('processCount');
    const loadingCount = document.getElementById('loadingCount');
    const processingModels = document.getElementById('processingModels');
    
    // Модели
    const modelCards = document.querySelectorAll('.model-card');
    const selectedModels = new Set();
    
    // Конфигурация API
//    const API_ENDPOINTS = {
//        vosk: 'https://bbahfftk2fo46cp2ivmf.containers.yandexcloud.net/transcribe',
//        nemo: 'https://bbae9hllahg5u86uh1nc.containers.yandexcloud.net/transcribe',
//        whisper: 'https://bba9l50oqhfrhaflmovj.containers.yandexcloud.net/transcribe'
//    };

    // Обработчики для выбора моделей
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            const model = this.getAttribute('data-model');
            
            if (selectedModels.has(model)) {
                selectedModels.delete(model);
                this.classList.remove('selected');
                // Удаляем счетчик
                const counter = this.querySelector('.selected-counter');
                if (counter) counter.remove();
            } else {
                selectedModels.add(model);
                this.classList.add('selected');
                // Добавляем счетчик
                const counter = document.createElement('div');
                counter.className = 'selected-counter';
                counter.textContent = Array.from(selectedModels).indexOf(model) + 1;
                this.appendChild(counter);
            }
            
            // Обновляем счетчики
            updateCounters();
            // Активируем кнопку, если есть выбранные модели и файл
            updateProcessButton();
        });
    });
    
    // Обработчики для загрузки файла
    browseBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#2ecc71';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.style.borderColor = '#3498db';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
    
    // Обработка выбора файла
    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) return;
        
        // Проверка типа файла
        if (!file.type.startsWith('audio/')) {
            fileInfo.innerHTML = '<p style="color: #e74c3c;">Пожалуйста, выберите аудиофайл</p>';
            return;
        }
        
        fileInfo.innerHTML = `<p>Выбран файл: <strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(2)} MB)</p>`;
        
        // Показать превью аудио
        const objectURL = URL.createObjectURL(file);
        audioPlayer.src = objectURL;
        audioPreview.style.display = 'block';
        
        // Активируем кнопку, если есть выбранные модели
        updateProcessButton();
    }
    
    // Обновление счетчиков
    function updateCounters() {
        const count = selectedModels.size;
        selectedCount.textContent = count;
        processCount.textContent = count;
        loadingCount.textContent = count;
    }
    
    // Обновление состояния кнопки обработки
    function updateProcessButton() {
        processBtn.disabled = !(selectedModels.size > 0 && fileInput.files.length > 0);
    }
    
    // Обработка нажатия кнопки "Запустить распознавание"
    processBtn.addEventListener('click', function() {
        if (processBtn.disabled) return;
        
        // Показать какие модели обрабатываются
        let modelsText = '';
        selectedModels.forEach(model => {
            const modelNames = {
                'whisper': 'OpenAI Whisper',
                'vosk': 'VOSK', 
                'nemo': 'NVIDIA Nemo'
            };
            modelsText += `<div style="margin: 5px 0;"><i class="fas fa-cog fa-spin"></i> ${modelNames[model]}</div>`;
        });
        processingModels.innerHTML = modelsText;
        
        // Показать индикатор загрузки
        loading.style.display = 'block';
        resultsSection.style.display = 'none';
        
        // Запускаем реальную обработку
        processAudioReal();
    });
    
    // Реальная обработка аудио
  
async function processAudioReal() {
    const file = fileInput.files[0];
    if (!file) return;

    resetAllResults();

    const promises = [];

    // Создаём промисы для каждой выбранной модели
    selectedModels.forEach(model => {
        if (API_ENDPOINTS[model]) {
            promises.push(sendToMicroserviceRaw(model, file)); 
        } else {
            promises.push(createDemoResult(model));
        }
    });

    try {
        const results = await Promise.allSettled(promises);
        results.forEach((result, index) => {
            const model = Array.from(selectedModels)[index];
            if (result.status === 'fulfilled') {
                updateResultCard(model, result.value);
            } else {
                updateResultCard(model, {
                    text: `Ошибка: ${result.reason.message || result.reason}`,
                    processing_time: '-',
                    confidence: '-',
                    error: true
                });
            }
        });
    } catch (error) {
        console.error('Ошибка при обработке:', error);
    } finally {
        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        showComparisonChart();
    }
}
    
    // Отправка запроса к микросервису
    async function sendToMicroserviceRaw(model, audioFile) {
        const startTime = Date.now();
    
        try {
            const arrayBuffer = await audioFile.arrayBuffer();
    
            const response = await fetch(API_ENDPOINTS[model], {
                method: 'POST',
                body: arrayBuffer,
                headers: {
                    'Content-Type': audioFile.type || 'audio/wav'
                }
            });
    
            if (!response.ok) {
                throw new Error(`Ошибка ${response.status}: ${await response.text()}`);
            }
    
            const result = await response.json();
            const endTime = Date.now();
    
            return {
                text: result.text || 'Не распознано',
                processing_time: result.processing_time || ((endTime - startTime) / 1000).toFixed(2) + 's',
                confidence: result.confidence ? result.confidence + '%' : '-',
                model: model
            };
        } catch (error) {
            console.error(`Ошибка ${model}:`, error);
            throw error;
        }
    }
    
    // Создание демо-результата для моделей без настроенного endpoint
    function createDemoResult(model) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const demoResults = {
                    whisper: {
                        text: "Это демо-текст от OpenAI Whisper. Модель показала высокую точность распознавания речи на русском языке.",
                        processing_time: (2 + Math.random()).toFixed(1) + 's',
                        confidence: (85 + Math.random() * 10).toFixed(1) + '%'
                    },
                    nemo: {
                        text: "Это демо-текст от NVIDIA Nemo. Модель демонстрирует хороший баланс между скоростью и точностью.",
                        processing_time: (1.5 + Math.random()).toFixed(1) + 's',
                        confidence: (80 + Math.random() * 15).toFixed(1) + '%'
                    }
                };
                
                resolve(demoResults[model] || {
                    text: "Демо-результат для выбранной модели",
                    processing_time: (1 + Math.random() * 2).toFixed(1) + 's',
                    confidence: (70 + Math.random() * 20).toFixed(1) + '%'
                });
            }, 2000 + Math.random() * 2000);
        });
    }
    
    // Обновление карточки с результатом
    function updateResultCard(model, result) {
        const resultElement = document.getElementById(`${model}Result`);
        resultElement.textContent = result.text;
        resultElement.classList.remove('empty');
        
        if (result.error) {
            resultElement.style.color = '#e74c3c';
        }
        
        document.getElementById(`${model}Time`).textContent = result.processing_time;
        document.getElementById(`${model}Confidence`).textContent = result.confidence;
        
        // Подсветить карточку с результатом
        const borderColor = 
            model === 'whisper' ? 'var(--whisper)' : 
            model === 'vosk' ? 'var(--vosk)' : 'var(--nemo)';
        document.getElementById(`${model}ResultCard`).style.borderColor = borderColor;
    }
    
    // Сброс всех результатов
    function resetAllResults() {
        const models = ['whisper', 'vosk', 'nemo'];
        models.forEach(model => {
            const resultElement = document.getElementById(`${model}Result`);
            resultElement.textContent = 'Эта модель не была выбрана для обработки';
            resultElement.classList.add('empty');
            resultElement.style.color = '';
            
            document.getElementById(`${model}Time`).textContent = '-';
            document.getElementById(`${model}Confidence`).textContent = '-';
            
            // Сбросить подсветку
            document.getElementById(`${model}ResultCard`).style.borderColor = 'var(--border)';
        });
    }
    
    // Показать сравнительную диаграмму
    function showComparisonChart() {
        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        // Данные для выбранных моделей
        const models = Array.from(selectedModels);
        const speeds = [];
        const confidences = [];
        const colors = [];
        const labels = [];
        
        models.forEach(model => {
            const timeElement = document.getElementById(`${model}Time`).textContent;
            const confidenceElement = document.getElementById(`${model}Confidence`).textContent;
            
            // Извлекаем числовые значения
            const speedValue = parseFloat(timeElement) || 0;
            const confidenceValue = parseFloat(confidenceElement) || 0;
            
            speeds.push(speedValue);
            confidences.push(confidenceValue);
            
            const color = 
                model === 'whisper' ? '#9b59b6' : 
                model === 'vosk' ? '#e74c3c' : '#f39c12';
            colors.push(color);
            
            const label = 
                model === 'whisper' ? 'Whisper' : 
                model === 'vosk' ? 'VOSK' : 'Nemo';
            labels.push(label);
        });
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Время (сек) - меньше лучше',
                        data: speeds,
                        backgroundColor: colors,
                        borderColor: colors,
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Уверенность (%) - больше лучше', 
                        data: confidences,
                        backgroundColor: colors.map(color => color + '80'),
                        borderColor: colors,
                        borderWidth: 1,
                        type: 'line',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Время (секунды)'
                        }
                    },
                    y1: {
                        position: 'right',
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Уверенность (%)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
    
    // Инициализация счетчиков
    updateCounters();
});
