import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/lib/api";
import { Upload, Loader2, CheckCircle, BarChart3 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function ImageUpload() {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [algorithm1, setAlgorithm1] = useState("metafed_hom");
  const [algorithm2, setAlgorithm2] = useState("metafed_het");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleCompare = async () => {
    if (!file) {
      toast({ title: "Error", description: "Please select a file first", variant: "destructive" });
      return;
    }

    setLoading(true);
    
    try {
      // Run both algorithms
      const [res1, res2] = await Promise.all([
        api.uploadImage(file, algorithm1),
        api.uploadImage(file, algorithm2),
      ]);
      
      setResults({
        [algorithm1]: res1,
        [algorithm2]: res2,
      });
      
      toast({ title: "Success", description: "Comparison complete!" });
    } catch (err) {
      // Demo results if backend not available
      setResults({
        [algorithm1]: {
          prediction: "Walking",
          confidence: 0.92,
          metrics: { accuracy: 0.879, precision: 0.872, recall: 0.876, f1: 0.874 }
        },
        [algorithm2]: {
          prediction: "Walking",
          confidence: 0.95,
          metrics: { accuracy: 0.891, precision: 0.885, recall: 0.888, f1: 0.886 }
        },
      });
      toast({ title: "Demo Mode", description: "Showing sample comparison results" });
    } finally {
      setLoading(false);
    }
  };

  const algorithmNames: Record<string, string> = {
    metafed_hom: "MetaFed (Homogeneous)",
    metafed_het: "MetaFed (Heterogeneous)",
  };

  const comparisonData = results ? [
    { metric: "Accuracy", [algorithmNames[algorithm1]]: results[algorithm1]?.metrics?.accuracy || 0, [algorithmNames[algorithm2]]: results[algorithm2]?.metrics?.accuracy || 0 },
    { metric: "Precision", [algorithmNames[algorithm1]]: results[algorithm1]?.metrics?.precision || 0, [algorithmNames[algorithm2]]: results[algorithm2]?.metrics?.precision || 0 },
    { metric: "Recall", [algorithmNames[algorithm1]]: results[algorithm1]?.metrics?.recall || 0, [algorithmNames[algorithm2]]: results[algorithm2]?.metrics?.recall || 0 },
    { metric: "F1 Score", [algorithmNames[algorithm1]]: results[algorithm1]?.metrics?.f1 || 0, [algorithmNames[algorithm2]]: results[algorithm2]?.metrics?.f1 || 0 },
  ] : [];

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Image Upload & Comparison</h1>
          <p className="text-muted-foreground">
            Upload sensor data and compare Homogeneous vs Heterogeneous MetaFed
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Data</CardTitle>
              <CardDescription>Upload sensor data file for activity prediction</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* File Upload */}
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                <input
                  type="file"
                  accept=".csv,.npy,.dat,image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-40 mx-auto rounded-lg mb-4" />
                  ) : (
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  )}
                  <p className="text-sm text-muted-foreground">
                    {file ? file.name : "Click to upload or drag and drop"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    CSV, NPY, DAT, or image files
                  </p>
                </label>
              </div>

              {/* Algorithm Selection */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Algorithm 1</label>
                  <Select value={algorithm1} onValueChange={setAlgorithm1}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="metafed_hom">MetaFed (Homogeneous)</SelectItem>
                      <SelectItem value="metafed_het">MetaFed (Heterogeneous)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Algorithm 2</label>
                  <Select value={algorithm2} onValueChange={setAlgorithm2}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="metafed_hom">MetaFed (Homogeneous)</SelectItem>
                      <SelectItem value="metafed_het">MetaFed (Heterogeneous)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button onClick={handleCompare} disabled={loading || !file} className="w-full">
                {loading ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</>
                ) : (
                  <><BarChart3 className="mr-2 h-4 w-4" /> Compare Algorithms</>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle>Comparison Results</CardTitle>
              <CardDescription>Prediction and metrics comparison</CardDescription>
            </CardHeader>
            <CardContent>
              {results ? (
                <div className="space-y-4">
                  {/* Predictions */}
                  <div className="grid grid-cols-2 gap-4">
                    {[algorithm1, algorithm2].map((algo) => (
                      <div key={algo} className="p-4 bg-muted/50 rounded-lg text-center">
                        <p className="text-xs text-muted-foreground mb-1">{algorithmNames[algo]}</p>
                        <p className="font-semibold text-lg">{results[algo]?.prediction}</p>
                        <Badge variant="outline" className="mt-2">
                          {(results[algo]?.confidence * 100).toFixed(1)}% confidence
                        </Badge>
                      </div>
                    ))}
                  </div>

                  {/* Comparison Chart */}
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={comparisonData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                        <YAxis domain={[0, 1]} />
                        <Tooltip formatter={(value: number) => (value * 100).toFixed(1) + "%"} />
                        <Legend />
                        <Bar dataKey={algorithmNames[algorithm1]} fill="#3b82f6" />
                        <Bar dataKey={algorithmNames[algorithm2]} fill="#22c55e" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Upload className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Upload a file and click Compare to see results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Algorithm Info */}
        <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-blue-500" />
                MetaFed (Homogeneous)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                All federations use the same CNN architecture. Knowledge is shared through 
                cyclic distillation, making it ideal when data distributions are similar.
              </p>
              <ul className="text-sm mt-3 space-y-1">
                <li>• Same model architecture (CNN) for all federations</li>
                <li>• Parameter averaging after local training</li>
                <li>• Better for homogeneous data distributions</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                MetaFed (Heterogeneous)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Different model architectures (CNN, RNN, ViT) per federation. Knowledge 
                distillation transfers features, enabling personalization.
              </p>
              <ul className="text-sm mt-3 space-y-1">
                <li>• Different architectures per federation</li>
                <li>• Feature-level knowledge distillation</li>
                <li>• Better for heterogeneous data distributions</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
