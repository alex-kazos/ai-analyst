// Simple scatter plot for clustering visualization using Plotly
function renderClusterScatterPlot(containerId, sampleData, xKey, yKey, clusterKey) {
    if (!sampleData || sampleData.length === 0) {
        document.getElementById(containerId).innerHTML = '<p class="text-muted">No data available for visualization.</p>';
        return;
    }
    const clusters = [...new Set(sampleData.map(d => d[clusterKey]))];
    const traces = clusters.map(cluster => {
        const clusterPoints = sampleData.filter(d => d[clusterKey] === cluster);
        return {
            x: clusterPoints.map(d => d[xKey]),
            y: clusterPoints.map(d => d[yKey]),
            mode: 'markers',
            type: 'scatter',
            name: 'Cluster ' + cluster,
            marker: { size: 10 }
        };
    });
    const layout = {
        title: 'Cluster Visualization',
        xaxis: { title: xKey },
        yaxis: { title: yKey },
        legend: { orientation: 'h', y: -0.2 },
        margin: { t: 40, l: 40, r: 10, b: 40 }
    };
    Plotly.newPlot(containerId, traces, layout, {responsive: true});
}
