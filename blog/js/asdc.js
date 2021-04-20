var margin = {top: 10, right: 30, bottom: 30, left: 40},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

var svg1 = d3.select("#eig_plot_1")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var svg2 = d3.select("#eig_plot_2")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

function initialize_svg(filename, svg, slider_selector) {
  d3.json(filename, function(json_data) {
    const A = math.matrix(json_data['A']);
    const B = math.matrix(json_data['B']);
    const Q = math.matrix(json_data['Q']);

    const n = A.size()[0];
    const m = 20;

    var eigs = [];
    var mu = 0;
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        eigs.push({
          'cc_weight': i / (m - 1),
          'eig_val': 0
        });
      }
    }
    eigenvalue_data(A, B, Q, mu, n, m, eigs);

    // Add X axis
    var x = d3.scaleLinear()
      .domain([0, 1])
      .range([ 0, width ]);
    var xAxis = svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

    // Add Y axis
    var y = d3.scaleLinear()
      .domain([-1.5, 1.5])
      .range([ height, 0]);
    svg.append("g")
      .call(d3.axisLeft(y));

    // Add dots
    svg.append('g')
    .selectAll("dot")
    .data(eigs)
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d['cc_weight']); } )
      .attr("cy", function (d) { return y(d['eig_val']); } )
      .attr("r", 5)
      .style("fill", "#69b3a2" )

    function updatePlot(new_mu) {
      mu = new_mu;
      eigenvalue_data(A, B, Q, mu, n, m, eigs);

      svg.selectAll("circle")
        .data(eigs)
        .transition()
        .duration(1000)
        .attr("cx", function (d) { return x(d['cc_weight']); } )
        .attr("cy", function (d) { return y(d['eig_val']); } )
    }

    var sliderSimple = d3
      .sliderBottom()
      .min(0)
      .max(1)
      .width(300)
      .tickFormat(d3.format('.2%'))
      .ticks(0)
      .default(0)
      .on('onchange', new_mu => {
        updatePlot(new_mu);
      });

    var gSimple = d3
      .select(slider_selector)
      .append('svg')
      .attr('width', 500)
      .attr('height', 100)
      .append('g')
      .attr('transform', 'translate(30,30)');

    gSimple.call(sliderSimple);

  });
}

initialize_svg("./js/augmenting_11.json", svg1, 'div#slider_cc_1');
initialize_svg("./js/augmenting_4.json", svg2, 'div#slider_cc_2');

function eigenvalue_data(A, B, Q, mu, n, m, eig_array) {
	Qmu = math.add(math.multiply(Q, math.sin(mu * math.pi / 2)), math.multiply(math.identity(n), math.cos(mu * math.pi / 2)));
	const conjA = math.multiply(math.transpose(Qmu), A, Qmu);
	const conjB = math.multiply(math.transpose(Qmu), B, Qmu);

	counter = 0;
	for (var i = 0; i < m; i ++) {
    var temp = math.add(math.multiply(1 - (i / (m - 1)), conjA), math.multiply(i / (m - 1), conjB));
    for (var j = 0; j < n; j++) {
      for (var k = j + 1; k < n; k++) {
        temp.set([j,k],temp.get([k,j]));
      }
    }
		const eigs = math.eigs(temp).values['_data'];
		for (var j = 0; j < eigs.length; j ++) {
			eig_array[counter].eig_val = eigs[j];
			counter ++;
		}
	}
}

