document.addEventListener("DOMContentLoaded", function() {
    var width = 600,
        height = 600,
        margin = 40;

    var radius = Math.min(width, height) / 2 - margin;

    var svg = d3.select("#chart3")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(" + (width / 2 + 30) + "," + height / 2 + ")"); // 右移 30px

    d3.csv("/static/csv/test-2.csv").then(function(data) {
        console.log("CSV data loaded:", data);

        var organizationTotalScores = d3.rollups(
            data,
            v => d3.sum(v, d => +d.total_score),
            d => d.organization
        );

        var totalScoreSum = d3.sum(organizationTotalScores, d => d[1]);

        var color = d3.scaleOrdinal()
            .domain(organizationTotalScores.map(function(d) { return d[0]; }))
            .range(d3.schemeCategory10);

        var pie = d3.pie()
            .value(function(d) { return d[1]; });

        var data_ready = pie(organizationTotalScores);

        var arc = d3.arc()
            .innerRadius(0)
            .outerRadius(radius);

        svg
            .selectAll('path')
            .data(data_ready)
            .enter()
            .append('path')
            .attr('d', arc)
            .attr('fill', function(d){ return(color(d.data[0])) })
            .attr("stroke", "white")
            .style("stroke-width", "2px")
            .style("opacity", 0.7);


        svg
            .selectAll('text')
            .data(data_ready)
            .enter()
            .append('text')
            .text(function(d){ 
                var percentage = (d.data[1] / totalScoreSum * 100).toFixed(2);
                return percentage + "%"; 
            })
            .attr("transform", function(d) { return "translate(" + arc.centroid(d) + ")"; })
            .style("text-anchor", "middle")
            .style("font-size", 30);

        // Add legend
        var legend = svg.append("g")
            .attr("transform", "translate(" + (-width / 2 + 10) + "," + (height / 2 - 150) + ")");

        legend.selectAll("rect")
            .data(data_ready)
            .enter()
            .append("rect")
            .attr("x", 10)
            .attr("y", function(d, i) { return i * 30+75; }) // 调整图例的垂直间距
            .attr("width", 10)
            .attr("height", 10)
            .attr("fill", function(d){ return color(d.data[0]); });

        legend.selectAll("text")
            .data(data_ready)
            .enter()
            .append("text")
            .attr("x", 30)
            .attr("y", function(d, i) { return i * 30 + 10+75; }) // 调整图例文字的垂直位置
            .text(function(d) { return d.data[0]; })
            .style("font-size", 15);
    }).catch(function(error) {
        console.error("Error loading the CSV file:", error);
    });
});
