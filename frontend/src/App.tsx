import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Dataset from "./pages/Dataset";
import Evaluation from "./pages/Evaluation";
// PersonalizedModels page removed
import ImageUpload from "./pages/ImageUpload";
import Comparison from "./pages/Comparison";
// BackendCode page removed
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/image-upload" element={<ImageUpload />} />
          <Route path="/comparison" element={<Comparison />} />
          {/* `/personalized` and `/backend-code` removed */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
