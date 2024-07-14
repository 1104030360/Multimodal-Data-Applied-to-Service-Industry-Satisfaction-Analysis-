
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
        //使用數據向SVG畫布中添加柱狀圖的矩形，設置其位置和大小。
        // 初始化柱狀圖，根據total_score設置顏色
        svg.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .style("fill", function(d) { 
                return d.total_score >= passed_score ? "green" : "red"; 
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

            function updateChart(scoreType) {
                svg.selectAll(".bar")
                    .transition()
                    .attr("y", function(d) { return y(d[scoreType]); })
                    .attr("height", function(d) { return height - y(d[scoreType]); })
                    .style("fill", function(d) { 
                        return d[scoreType] >= passed_score ? "green" : "red"; 
                    });
            
                svg.selectAll(".bar-label")
                    .transition()
                    .attr("y", function(d) { return y(d[scoreType]) - 5; })
                    .text(function(d) { return d[scoreType]; });
            }
            
            

        d3.select("#facial_score_btn").on("click", function() { updateChart("facial_score"); });
        d3.select("#audio_score_btn").on("click", function() { updateChart("audio_score"); });
        d3.select("#text_score_btn").on("click", function() { updateChart("text_score"); });
        d3.select("#total_score_btn").on("click", function() { updateChart("total_score"); });
    }).catch(function(error) {
        console.error("Error loading the CSV file:", error);
    });
});
