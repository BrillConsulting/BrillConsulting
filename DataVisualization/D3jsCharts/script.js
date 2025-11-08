// Tab switching functionality
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        const tabId = button.getAttribute('data-tab');

        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
    });
});

// ========== FORCE-DIRECTED GRAPH ==========
function createForceDirectedGraph() {
    const width = document.getElementById('force-svg').clientWidth;
    const height = 600;

    // Sample data
    const nodes = Array.from({length: 30}, (_, i) => ({
        id: `node${i}`,
        group: Math.floor(Math.random() * 5) + 1
    }));

    const links = Array.from({length: 50}, () => ({
        source: `node${Math.floor(Math.random() * 30)}`,
        target: `node${Math.floor(Math.random() * 30)}`,
        value: Math.random() * 10
    }));

    const svg = d3.select('#force-svg');
    svg.selectAll('*').remove();

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));

    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => Math.sqrt(d.value));

    const node = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    node.append('circle')
        .attr('r', 15)
        .attr('fill', d => d3.schemeCategory10[d.group % 10]);

    const labels = node.append('text')
        .text(d => d.id)
        .attr('x', 20)
        .attr('y', 5)
        .style('display', 'none');

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    // Controls
    document.getElementById('reset-force').onclick = () => {
        simulation.alpha(1).restart();
    };

    document.getElementById('toggle-labels').onclick = () => {
        const currentDisplay = labels.style('display');
        labels.style('display', currentDisplay === 'none' ? 'block' : 'none');
    };
}

// ========== ANIMATED BAR CHART ==========
function createBarChart() {
    const width = document.getElementById('bar-svg').clientWidth;
    const height = 600;
    const margin = {top: 40, right: 30, bottom: 60, left: 60};

    let data = Array.from({length: 10}, (_, i) => ({
        name: `Item ${String.fromCharCode(65 + i)}`,
        value: Math.floor(Math.random() * 100) + 10
    }));

    const svg = d3.select('#bar-svg');
    svg.selectAll('*').remove();

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
        .range([0, width - margin.left - margin.right])
        .padding(0.2);

    const y = d3.scaleLinear()
        .range([height - margin.top - margin.bottom, 0]);

    const xAxis = g.append('g')
        .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
        .attr('class', 'x-axis');

    const yAxis = g.append('g')
        .attr('class', 'y-axis');

    function update(newData) {
        x.domain(newData.map(d => d.name));
        y.domain([0, d3.max(newData, d => d.value) * 1.1]);

        xAxis.transition().duration(750).call(d3.axisBottom(x));
        yAxis.transition().duration(750).call(d3.axisLeft(y));

        const bars = g.selectAll('.bar')
            .data(newData, d => d.name);

        bars.enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => x(d.name))
            .attr('width', x.bandwidth())
            .attr('y', height - margin.top - margin.bottom)
            .attr('height', 0)
            .attr('fill', '#667eea')
            .merge(bars)
            .transition()
            .duration(750)
            .attr('x', d => x(d.name))
            .attr('width', x.bandwidth())
            .attr('y', d => y(d.value))
            .attr('height', d => height - margin.top - margin.bottom - y(d.value));

        bars.exit()
            .transition()
            .duration(750)
            .attr('y', height - margin.top - margin.bottom)
            .attr('height', 0)
            .remove();

        // Value labels
        const labels = g.selectAll('.value-label')
            .data(newData, d => d.name);

        labels.enter()
            .append('text')
            .attr('class', 'value-label')
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#333')
            .merge(labels)
            .transition()
            .duration(750)
            .attr('x', d => x(d.name) + x.bandwidth() / 2)
            .attr('y', d => y(d.value) - 5)
            .text(d => d.value);

        labels.exit().remove();
    }

    update(data);

    // Controls
    document.getElementById('sort-bars').onclick = () => {
        data.sort((a, b) => b.value - a.value);
        update(data);
    };

    document.getElementById('randomize-bars').onclick = () => {
        data.forEach(d => d.value = Math.floor(Math.random() * 100) + 10);
        update(data);
    };

    document.getElementById('add-bar').onclick = () => {
        const newLetter = String.fromCharCode(65 + data.length);
        data.push({name: `Item ${newLetter}`, value: Math.floor(Math.random() * 100) + 10});
        update(data);
    };
}

// ========== TREEMAP ==========
function createTreemap() {
    const width = document.getElementById('treemap-svg').clientWidth;
    const height = 600;

    const data = {
        name: 'Root',
        children: [
            {
                name: 'Technology',
                children: [
                    {name: 'AI/ML', value: 1500},
                    {name: 'Cloud', value: 1200},
                    {name: 'Data', value: 1100},
                    {name: 'Security', value: 900}
                ]
            },
            {
                name: 'Business',
                children: [
                    {name: 'Finance', value: 1300},
                    {name: 'Marketing', value: 1100},
                    {name: 'Sales', value: 950},
                    {name: 'HR', value: 700}
                ]
            },
            {
                name: 'Operations',
                children: [
                    {name: 'Supply Chain', value: 1000},
                    {name: 'Logistics', value: 850},
                    {name: 'Manufacturing', value: 1250}
                ]
            }
        ]
    };

    const svg = d3.select('#treemap-svg');
    svg.selectAll('*').remove();

    const root = d3.hierarchy(data)
        .sum(d => d.value)
        .sort((a, b) => b.value - a.value);

    d3.treemap()
        .size([width, height])
        .padding(2)
        .round(true)
        (root);

    const color = d3.scaleOrdinal(d3.schemeSet3);

    const cell = svg.selectAll('g')
        .data(root.leaves())
        .join('g')
        .attr('transform', d => `translate(${d.x0},${d.y0})`);

    cell.append('rect')
        .attr('class', 'treemap-cell')
        .attr('width', d => d.x1 - d.x0)
        .attr('height', d => d.y1 - d.y0)
        .attr('fill', (d, i) => color(i));

    cell.append('text')
        .attr('class', 'treemap-text')
        .attr('x', d => (d.x1 - d.x0) / 2)
        .attr('y', d => (d.y1 - d.y0) / 2)
        .text(d => d.data.name)
        .style('font-size', d => Math.min((d.x1 - d.x0) / 6, (d.y1 - d.y0) / 3, 14) + 'px');
}

// ========== SUNBURST ==========
function createSunburst() {
    const width = document.getElementById('sunburst-svg').clientWidth;
    const height = 600;
    const radius = Math.min(width, height) / 2;

    const data = {
        name: 'Portfolio',
        children: [
            {
                name: 'Data Science',
                children: [
                    {name: 'ML', value: 15},
                    {name: 'Statistics', value: 12},
                    {name: 'Analytics', value: 10}
                ]
            },
            {
                name: 'Engineering',
                children: [
                    {name: 'Backend', value: 14},
                    {name: 'Frontend', value: 11},
                    {name: 'DevOps', value: 9}
                ]
            },
            {
                name: 'Cloud',
                children: [
                    {name: 'AWS', value: 10},
                    {name: 'Azure', value: 9},
                    {name: 'GCP', value: 8}
                ]
            }
        ]
    };

    const svg = d3.select('#sunburst-svg');
    svg.selectAll('*').remove();

    const g = svg.append('g')
        .attr('transform', `translate(${width/2},${height/2})`);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const partition = d3.partition()
        .size([2 * Math.PI, radius]);

    const arc = d3.arc()
        .startAngle(d => d.x0)
        .endAngle(d => d.x1)
        .innerRadius(d => d.y0)
        .outerRadius(d => d.y1);

    const root = d3.hierarchy(data)
        .sum(d => d.value);

    partition(root);

    g.selectAll('path')
        .data(root.descendants())
        .join('path')
        .attr('class', 'arc')
        .attr('d', arc)
        .attr('fill', d => color((d.children ? d : d.parent).data.name))
        .on('click', clicked);

    function clicked(event, p) {
        root.each(d => d.target = {
            x0: Math.max(0, Math.min(1, (d.x0 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
            x1: Math.max(0, Math.min(1, (d.x1 - p.x0) / (p.x1 - p.x0))) * 2 * Math.PI,
            y0: Math.max(0, d.y0 - p.y0),
            y1: Math.max(0, d.y1 - p.y0)
        });

        const t = g.transition().duration(750);

        g.selectAll('path')
            .transition(t)
            .tween('data', d => {
                const i = d3.interpolate(d.current, d.target);
                return t => d.current = i(t);
            })
            .attr('d', d => arc(d.current));
    }

    root.each(d => d.current = {x0: d.x0, x1: d.x1, y0: d.y0, y1: d.y1});
}

// ========== CHORD DIAGRAM ==========
function createChordDiagram() {
    const width = document.getElementById('chord-svg').clientWidth;
    const height = 600;
    const outerRadius = Math.min(width, height) * 0.4;
    const innerRadius = outerRadius - 30;

    // Matrix representing relationships
    const matrix = [
        [0, 5, 6, 4, 7, 4],
        [5, 0, 5, 4, 6, 3],
        [6, 5, 0, 4, 5, 5],
        [4, 4, 4, 0, 5, 4],
        [7, 6, 5, 5, 0, 4],
        [4, 3, 5, 4, 4, 0]
    ];

    const names = ['Python', 'JavaScript', 'SQL', 'R', 'Java', 'Go'];
    const colors = d3.schemeCategory10;

    const svg = d3.select('#chord-svg');
    svg.selectAll('*').remove();

    const g = svg.append('g')
        .attr('transform', `translate(${width/2},${height/2})`);

    const chord = d3.chord()
        .padAngle(0.05)
        .sortSubgroups(d3.descending);

    const arc = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius);

    const ribbon = d3.ribbon()
        .radius(innerRadius);

    const chords = chord(matrix);

    const group = g.append('g')
        .selectAll('g')
        .data(chords.groups)
        .join('g');

    group.append('path')
        .attr('fill', d => colors[d.index])
        .attr('stroke', d => d3.rgb(colors[d.index]).darker())
        .attr('d', arc);

    group.append('text')
        .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
        .attr('dy', '.35em')
        .attr('transform', d => `
            rotate(${(d.angle * 180 / Math.PI - 90)})
            translate(${outerRadius + 10})
            ${d.angle > Math.PI ? 'rotate(180)' : ''}
        `)
        .attr('text-anchor', d => d.angle > Math.PI ? 'end' : null)
        .text(d => names[d.index])
        .style('font-size', '12px')
        .style('font-weight', 'bold');

    g.append('g')
        .attr('fill-opacity', 0.67)
        .selectAll('path')
        .data(chords)
        .join('path')
        .attr('class', 'chord')
        .attr('d', ribbon)
        .attr('fill', d => colors[d.target.index])
        .attr('stroke', d => d3.rgb(colors[d.target.index]).darker())
        .on('mouseover', function(event, d) {
            d3.select(this).attr('fill-opacity', 1);
            document.getElementById('chord-info').textContent =
                `${names[d.source.index]} → ${names[d.target.index]}: ${d.source.value}`;
        })
        .on('mouseout', function() {
            d3.select(this).attr('fill-opacity', 0.67);
            document.getElementById('chord-info').textContent = 'Hover over chords to see relationships';
        });
}

// Initialize all visualizations
createForceDirectedGraph();
createBarChart();
createTreemap();
createSunburst();
createChordDiagram();

// Handle window resize
window.addEventListener('resize', () => {
    createForceDirectedGraph();
    createBarChart();
    createTreemap();
    createSunburst();
    createChordDiagram();
});
