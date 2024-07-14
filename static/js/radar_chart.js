document.addEventListener("DOMContentLoaded", function() {
    var margin = { top: 60, right: 80, bottom: 50, left: 80 },
        width = Math.min(700, window.innerWidth - 10) - margin.left - margin.right,
        height = Math.min(width, window.innerHeight - margin.top - margin.bottom - 20);

    var radarChartOptions = {
        w: width,
        h: height,
        margin: margin,
        levels: 5,
        roundStrokes: true,
        color: d3.scaleOrdinal(d3.schemeCategory10)
    };

    // Data
    d3.csv("/static/csv/test-2.csv").then(function(data) {
        var radarData = data.map(d => ({
            name: d.name,
            axes: [
                { axis: 'Facial Score', value: +d.facial_score },
                { axis: 'Audio Score', value: +d.audio_score },
                { axis: 'Text Score', value: +d.text_score }
            ]
        }));

        RadarChart("#chart2", radarData, radarChartOptions);
    });

    // RadarChart function
    function RadarChart(id, data, options) {
        var cfg = {
            w: 600,
            h: 600,
            margin: { top: 20, right: 20, bottom: 20, left: 20 },
            levels: 3,
            maxValue: 100, // 确保最大值为100
            labelFactor: 1.25,
            wrapWidth: 60,
            opacityArea: 0.35,
            dotRadius: 4,
            opacityCircles: 0.1,
            strokeWidth: 2,
            roundStrokes: false,
            color: d3.scaleOrdinal(d3.schemeCategory10)
        };

        if ('undefined' !== typeof options) {
            for (var i in options) {
                if ('undefined' !== typeof options[i]) { cfg[i] = options[i]; }
            }
        }

        cfg.maxValue = Math.max(cfg.maxValue, d3.max(data, function(i) {
            return d3.max(i.axes, function(o) { return o.value; });
        }));

        var allAxis = (data[0].axes.map(function(i, j) { return i.axis; })),
            total = allAxis.length,
            radius = Math.min(cfg.w / 2, cfg.h / 2),
            angleSlice = Math.PI * 2 / total;

        var rScale = d3.scaleLinear()
            .range([0, radius])
            .domain([0, 100]); // 确保最大值为100

        var svg = d3.select(id).append("svg")
            .attr("width", cfg.w + cfg.margin.left + cfg.margin.right)
            .attr("height", cfg.h + cfg.margin.top + cfg.margin.bottom)
            .attr("class", "radar" + id);

        var g = svg.append("g")
            .attr("transform", "translate(" + (cfg.w / 2 + cfg.margin.left) + "," + (cfg.h / 2 + cfg.margin.top) + ")");

        var filter = g.append('defs').append('filter').attr('id', 'glow'),
            feGaussianBlur = filter.append('feGaussianBlur').attr('stdDeviation', '2.5').attr('result', 'coloredBlur'),
            feMerge = filter.append('feMerge'),
            feMergeNode_1 = feMerge.append('feMergeNode').attr('in', 'coloredBlur'),
            feMergeNode_2 = feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        var axisGrid = g.append("g").attr("class", "axisWrapper");

        axisGrid.selectAll(".levels")
            .data(d3.range(1, (cfg.levels + 1)).reverse())
            .enter().append("circle")
            .attr("class", "gridCircle")
            .attr("r", function(d, i) { return radius / cfg.levels * d; })
            .style("fill", "#CDCDCD")
            .style("stroke", "#CDCDCD")
            .style("fill-opacity", cfg.opacityCircles)
            .style("filter", "url(#glow)");

        axisGrid.selectAll(".axisLabel")
            .data(d3.range(1, (cfg.levels + 1)).reverse())
            .enter().append("text")
            .attr("class", "axisLabel")
            .attr("x", 4)
            .attr("y", function(d) { return -d * radius / cfg.levels; })
            .attr("dy", "0.4em")
            .style("font-size", "15px")
            .attr("fill", "#737373")
            .text(function(d, i) { return (d * 100 / cfg.levels) + "%"; }); // 确保百分比格式正确

        var axis = axisGrid.selectAll(".axis")
            .data(allAxis)
            .enter().append("g")
            .attr("class", "axis");

        axis.append("line")
            .attr("x1", 0)
            .attr("y1", 0)
            .attr("x2", function(d, i) { return rScale(100) * Math.cos(angleSlice * i - Math.PI / 2); })
            .attr("y2", function(d, i) { return rScale(100) * Math.sin(angleSlice * i - Math.PI / 2); })
            .attr("class", "line")
            .style("stroke", "white")
            .style("stroke-width", "2px");

        axis.append("text")
            .attr("class", "legend")
            .style("font-size", "15px")
            .attr("text-anchor", "middle")
            .attr("dy", "1.5em")
            .attr("x", function(d, i) { return rScale(100 * cfg.labelFactor) * Math.cos(angleSlice * i - Math.PI / 2); })
            .attr("y", function(d, i) { return rScale(100 * cfg.labelFactor) * Math.sin(angleSlice * i - Math.PI / 2); })
            .text(function(d) { return d; })
            .call(wrap, cfg.wrapWidth);

        var radarLine = d3.lineRadial()
            .curve(d3.curveLinearClosed)
            .radius(function(d) { return rScale(d.value); })
            .angle(function(d, i) { return i * angleSlice; });

        if (cfg.roundStrokes) {
            radarLine.curve(d3.curveCardinalClosed);
        }

        var blobWrapper = g.selectAll(".radarWrapper")
            .data(data)
            .enter().append("g")
            .attr("class", "radarWrapper");

        blobWrapper.append("path")
            .attr("class", "radarStroke")
            .attr("d", function(d, i) { return radarLine(d.axes); })
            .style("stroke-width", cfg.strokeWidth + "px")
            .style("stroke", function(d, i) { return cfg.color(i); })
            .style("fill", "none")
            .style("filter", "url(#glow)");

        blobWrapper.selectAll(".radarCircle")
            .data(function(d, i) { return d.axes; })
            .enter().append("circle")
            .attr("class", "radarCircle")
            .attr("r", cfg.dotRadius)
            .attr("cx", function(d, i) { return rScale(d.value) * Math.cos(angleSlice * i - Math.PI / 2); })
            .attr("cy", function(d, i) { return rScale(d.value) * Math.sin(angleSlice * i - Math.PI / 2); })
            .style("fill", function(d, i, j) { return cfg.color(j); })
            .style("fill-opacity", 0.8);

        // 添加图例
        var legend = svg.append("g")
            .attr("class", "legendWrapper")
            .attr("transform", "translate(" + (cfg.w / 2 + cfg.margin.left) + "," + (cfg.h + cfg.margin.top + 20) + ")");

        legend.selectAll(".legendSquare")
            .data(data)
            .enter().append("rect")
            .attr("class", "legendSquare")
            .attr("x", function(d, i) { return i * 80; })
            .attr("y", 0)
            .attr("width", 10)
            .attr("height", 10)
            .style("fill", function(d, i) { return cfg.color(i); });

        legend.selectAll(".legendText")
            .data(data)
            .enter().append("text")
            .attr("class", "legendText")
            .attr("x", function(d, i) { return i * 80 + 15; })
            .attr("y", 10)
            .style("font-size", "15px")
            .attr("fill", "#737373")
            .text(function(d) { return d.name; });

        function wrap(text, width) {
            text.each(function() {
                var text = d3.select(this),
                    words = text.text().split(/\s+/).reverse(),
                    word,
                    line = [], lineNumber = 0,
                    lineHeight = 1.4,
                    y = text.attr("y"),
                    x = text.attr("x"),
                    dy = parseFloat(text.attr("dy")),
                    tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy + "em");

                while (word = words.pop()) {
                    line.push(word);
                    tspan.text(line.join(" "));
                    if (tspan.node().getComputedTextLength() > width) {
                        line.pop();
                        tspan.text(line.join(" "));
                        line = [word];
                        tspan = text.append("tspan").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
                    }
                }
            });
        }
    }
});
