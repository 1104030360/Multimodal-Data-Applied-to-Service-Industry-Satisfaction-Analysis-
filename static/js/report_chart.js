
//當頁面加載完成時，執行內部的函數。

var passed_score = 70

document.addEventListener("DOMContentLoaded", function() {
    var margin = {top: 30, right: 30, bottom: 70, left: 40},
        width = 1200 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    var x = d3.scaleBand().rangeRound([0, width]).padding(0.5);
    var y = d3.scaleLinear().range([height, 0]);

    //定義X軸和Y軸，X軸在底部顯示，Y軸在左側顯示，Y軸顯示10個刻度。
    var xAxis = d3.axisBottom(x);
    var yAxis = d3.axisLeft(y).ticks(5);

    var svg = d3.select("#chart")
        .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    var passed_score = 70
    


    //使用D3的csv方法讀取CSV文件，讀取完成後執行內部函數並打印數據
    d3.csv("/static/csv/test-2.csv").then(function(data) {
        console.log("CSV data loaded:", data);
        //將讀取的數據轉換為數字類型
        data.forEach(function(d) {
            d.facial_score = +d.facial_score;
            d.audio_score = +d.audio_score;
            d.text_score = +d.text_score;
            d.total_score = +d.total_score;
        });

        x.domain(data.map(function(d) { return d.name; }));
        y.domain([0,100]);
        //向SVG畫布中添加X軸，並旋轉X軸上的文字標籤。
        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis)
            .selectAll("text")
            .style("text-anchor", "end")
            .style("font-family", "Arial")
            .style("font-size", "20px") // 設置字體大小
            .attr("dx", "1.4em")
            .attr("dy", "1em")
            .attr("transform", "rotate(0)");
        //向SVG畫布中添加Y軸。
        svg.append("g")
            .attr("class", "y axis")
            .style("text-anchor", "end")
            .style("font-family", "Arial")
            .style("font-size", "20px") // 設置字體大小
            .call(yAxis);


        var initialBaselineValue = d3.mean(data, function(d) { return d.total_score; });

        var baseline = svg.append("line")
        // Add a baseline            
            .attr("x1", 0)
            .attr("x2", width)
            .attr("y1", y(initialBaselineValue))
            .attr("y2", y(initialBaselineValue))
            .attr("stroke-width", 2)
            .attr("stroke", "gray")
            .attr("stroke-dasharray", "5,5");
 
 
        //使用數據向SVG畫布中添加柱狀圖的矩形，設置其位置和大小。
        // 初始化柱狀圖，根據total_score設置顏色
        svg.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .style("fill", function(d) { 
                return d.total_score >= initialBaselineValue ? "green" : "red"; 
            })
            .attr("x", function(d) { return x(d.name); })
            .attr("width", x.bandwidth())
            .attr("y", function(d) { return y(d.total_score); })
            .attr("height", function(d) { return height - y(d.total_score); });

        svg.selectAll(".bar-label")
            .data(data)
            .enter().append("text")
            .attr("class", "bar-label")
            .attr("x", function(d) { return x(d.name) + x.bandwidth() / 2; })
            .attr("y", function(d) { return y(d.total_score) - 5; })
            .attr("text-anchor", "middle")
            .style("font-family", "Arial")
            .style("font-size", "15px")
            .text(function(d) { return d.total_score; });

            
        // 更新基準線函數
        function updateBaseline(scoreType) {
            var averageScore = d3.mean(data, d => d[scoreType]);
            baseline.attr("x1", 0)
                    .attr("x2", width)
                    .attr("y1", y(averageScore))
                    .attr("y2", y(averageScore));
            return averageScore;  // 返回當前平均分數以供後續比較

        }



            function setupButtons() {
                document.getElementById('facial_score_btn').addEventListener('click', function() {
                    updateChart('facial_score');
                    updateSuggestion('summarize_text1');
                });
                document.getElementById('audio_score_btn').addEventListener('click', function() {
                    updateChart('audio_score');
                    updateSuggestion('summarize_text2');
                });
                document.getElementById('text_score_btn').addEventListener('click', function() {
                    updateChart('text_score');
                    updateSuggestion('summarize_text3');
                });
                document.getElementById('total_score_btn').addEventListener('click', function() {
                    updateChart('total_score');
                    updateSuggestion('summarize_text4');
                });
            }

            function updateChart(scoreType) {
                var averageScore = updateBaseline(scoreType);  // Update baseline and get average score

                svg.selectAll(".bar")
                    .transition()
                    .attr("y", function(d) { return y(d[scoreType]); })
                    .attr("height", function(d) { return height - y(d[scoreType]); })
                    .style("fill", function(d) { 
                        return d[scoreType] >= averageScore ? "green" : "red"; 
                    });
            
                svg.selectAll(".bar-label")
                    .transition()
                    .attr("y", function(d) { return y(d[scoreType]) - 5; })
                    .text(function(d) { return d[scoreType]; });
                updateBaseline(scoreType)
            }

            var typingTimeout; // Global variable to keep track of the timeout

            function updateSuggestion(scoreType) {
                fetch(`/api/get_ai_suggestion?type=${scoreType}`) // Fetch the API
                .then(response => response.json())
                .then(data => {
                    typeWriter('ai_suggestion_text', data.suggestion, 50);
                })
                .catch(error => console.error('Error fetching AI suggestions:', error));
            }
            
            function typeWriter(elementId, text, speed) {
                const elem = document.getElementById(elementId);
                if (typingTimeout) {
                    clearTimeout(typingTimeout);  // Clear the existing timeout if there is one
                    elem.innerText = ''; // Immediately clear previous text
                }
                let i = 0;
                elem.innerText = ''; // Clear the element's text before starting new typing
            
                function typing() {
                    if (i < text.length) {
                        elem.innerHTML += text.charAt(i);
                        i++;
                        typingTimeout = setTimeout(typing, speed); // Update the timeout variable
                    }
                }
                typing(); // Start the typing effect
            }
            

            

            
            
            

        d3.select("#facial_score_btn").on("click", function() { 
            updateChart("facial_score")
            updateSuggestion('summarize_text1');
         });
        d3.select("#audio_score_btn").on("click", function() { 
            updateChart("audio_score"); 
            updateSuggestion('summarize_text2');});
        d3.select("#text_score_btn").on("click", function() { 
            updateChart("text_score");
            updateSuggestion('summarize_text3');
         });
        d3.select("#total_score_btn").on("click", function() { 
            updateChart("total_score")
            updateSuggestion('summarize_text4'); 
        });
        





    }).catch(function(error) {
        console.error("Error loading the CSV file:", error);
    });
});
