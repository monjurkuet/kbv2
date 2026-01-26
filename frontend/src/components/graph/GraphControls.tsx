import { createSignal, Show } from 'solid-js';
import type { Sigma as SigmaType } from 'sigma';

interface GraphControlsProps {
  sigma?: SigmaType;
}

const GraphControls = (props: GraphControlsProps) => {
  const [isZoomed, setIsZoomed] = createSignal(false);

  const handleZoomIn = () => {
    if (props.sigma) {
      const camera = props.sigma.getCamera();
      camera.setState({ ...camera.getState(), ratio: camera.getState().ratio * 0.8 });
    }
  };

  const handleZoomOut = () => {
    if (props.sigma) {
      const camera = props.sigma.getCamera();
      camera.setState({ ...camera.getState(), ratio: camera.getState().ratio * 1.25 });
    }
  };

  const handleFit = () => {
    if (props.sigma) {
      props.sigma.getCamera().animatedReset({ duration: 300 });
      setIsZoomed(false);
    }
  };

  const handleToggleLayout = () => {
    setIsZoomed(!isZoomed());
  };

  return (
    <div class="absolute top-4 left-4 bg-white rounded-lg shadow-lg p-2 space-y-2" data-testid="graph-controls">
      <button
        onClick={handleZoomIn}
        class="w-full px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors flex items-center justify-center"
        title="Zoom In"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
        </svg>
      </button>

      <button
        onClick={handleZoomOut}
        class="w-full px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors flex items-center justify-center"
        title="Zoom Out"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7"></path>
        </svg>
      </button>

      <button
        onClick={handleFit}
        class="w-full px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors flex items-center justify-center"
        title="Fit to View"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path>
        </svg>
      </button>

      <div class="border-t border-gray-200 pt-2">
        <button
          onClick={handleToggleLayout}
          class={`w-full px-3 py-2 text-sm font-medium rounded-md transition-colors flex items-center justify-center ${
            isZoomed()
              ? 'text-blue-700 bg-blue-100 hover:bg-blue-200'
              : 'text-gray-700 bg-gray-100 hover:bg-gray-200'
          }`}
          title="Toggle Layout"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default GraphControls;