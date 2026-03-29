import { useState } from "react";
import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { api, ALGORITHM_NAMES, METRICS } from "@/lib/api";
import { Play, Loader2, BarChart3, TrendingUp } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";

// Sample data for demo (will be replaced by API data)
const sampleHomogeneousResults = {
  fedavg: { accuracy: 0.823, precision: 0.815, recall: 0.820, f1: 0.817 },
  fedprox: { accuracy: 0.841, precision: 0.835, recall: 0.838, f1: 0.836 },
  fedbn: { accuracy: 0.856, precision: 0.849, recall: 0.853, f1: 0.851 },
  metafed_hom: { accuracy: 0.879, precision: 0.872, recall: 0.876, f1: 0.874 },
};

const sampleHeterogeneousResults = {
  fed_0: { accuracy: 0.891, precision: 0.885, recall: 0.888, f1: 0.886, model: "CNN" },
  fed_1: { accuracy: 0.867, precision: 0.861, recall: 0.864, f1: 0.862, model: "RNN" },
  fed_2: { accuracy: 0.854, precision: 0.848, recall: 0.851, f1: 0.849, model: "ViT" },
};

export default function Evaluation() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [homResults, setHomResults] = useState(sampleHomogeneousResults);
  const [hetResults, setHetResults] = useState(sampleHeterogeneousResults);
  const [experimentRun, setExperimentRun] = useState(false);

  const runHomogeneous = async () => {
    setLoading(true);
    setProgress(0);
    
    // Simulate progress
    const interval = setInterval(() => {
      setProgress((p) => Math.min(p + 10, 90));
    }, 500);

    try {
      const data = await api.runHomogeneous();
      setHomResults(data.results || sampleHomogeneousResults);
      setExperimentRun(true);
      toast({ title: "Success", description: "Homogeneous experiments completed!" });
    } catch (err) {
      toast({ title: "Using Demo Data", description: "Backend not connected. Showing sample results." });
      setExperimentRun(true);
    } finally {
      clearInterval(interval);
      setProgress(100);
      setLoading(false);
    }
  };

  const runHeterogeneous = async () => {
    setLoading(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress((p) => Math.min(p + 10, 90));
    }, 500);

    try {
      const data = await api.runHeterogeneous();
      setHetResults(data.results || sampleHeterogeneousResults);
      setExperimentRun(true);
      toast({ title: "Success", description: "Heterogeneous MetaFed completed!" });
    } catch (err) {
      toast({ title: "Using Demo Data", description: "Backend not connected. Showing sample results." });
      setExperimentRun(true);
    } finally {
      clearInterval(interval);
      setProgress(100);
      setLoading(false);
    }
  };

  // Prepare chart data
  const barChartData = METRICS.map((metric) => ({
    metric: metric.charAt(0).toUpperCase() + metric.slice(1),
    FedAvg: homResults.fedavg[metric],
    FedProx: homResults.fedprox[metric],
    FedBN: homResults.fedbn[metric],
    MetaFed: homResults.metafed_hom[metric],
  }));

  const radarData = Object.entries(homResults).map(([algo, metrics]) => ({
    algorithm: ALGORITHM_NAMES[algo] || algo,
    ...metrics,
  }));

  const hetBarData = METRICS.map((metric) => ({
    metric: metric.charAt(0).toUpperCase() + metric.slice(1),
    "Fed 0 (CNN)": hetResults.fed_0[metric],
    "Fed 1 (RNN)": hetResults.fed_1[metric],
    "Fed 2 (ViT)": hetResults.fed_2[metric],
  }));

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Evaluation Metrics</h1>
          <p className="text-muted-foreground">Run experiments and analyze results</p>
        </div>

        {/* Run Buttons */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          <Button onClick={runHomogeneous} disabled={loading} size="lg">
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run Homogeneous FL (CNN)
          </Button>
          <Button onClick={runHeterogeneous} disabled={loading} size="lg" variant="outline">
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Run Heterogeneous MetaFed
          </Button>
        </div>

        {loading && (
          <Card className="max-w-md mx-auto mb-8">
            <CardContent className="pt-6">
              <p className="text-center mb-2">Training in progress...</p>
              <Progress value={progress} className="h-2" />
              <p className="text-center text-sm text-muted-foreground mt-2">{progress}%</p>
            </CardContent>
          </Card>
        )}

        <Tabs defaultValue="homogeneous" className="max-w-6xl mx-auto">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="homogeneous">Homogeneous FL (CNN)</TabsTrigger>
            <TabsTrigger value="heterogeneous">Heterogeneous MetaFed</TabsTrigger>
          </TabsList>

          <TabsContent value="homogeneous">
            {/* Metrics Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              {Object.entries(homResults).map(([algo, metrics]) => (
                <Card key={algo}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">{ALGORITHM_NAMES[algo] || algo}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold text-primary">
                      {(metrics.accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground">Accuracy</p>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Bar Chart */}
            <Card className="mb-8">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Algorithm Comparison
                </CardTitle>
                <CardDescription>All metrics across FL algorithms</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={barChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip formatter={(value: number) => (value * 100).toFixed(1) + "%"} />
                    <Legend />
                    <Bar dataKey="FedAvg" fill="#3b82f6" />
                    <Bar dataKey="FedProx" fill="#f97316" />
                    <Bar dataKey="FedBN" fill="#22c55e" />
                    <Bar dataKey="MetaFed" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Detailed Table */}
            <Card>
              <CardHeader>
                <CardTitle>Detailed Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-3">Algorithm</th>
                        {METRICS.map((m) => (
                          <th key={m} className="text-left p-3">{m.charAt(0).toUpperCase() + m.slice(1)}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(homResults).map(([algo, metrics]) => (
                        <tr key={algo} className="border-b hover:bg-muted/50">
                          <td className="p-3 font-medium">{ALGORITHM_NAMES[algo] || algo}</td>
                          {METRICS.map((m) => (
                            <td key={m} className="p-3">{(metrics[m] * 100).toFixed(2)}%</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="heterogeneous">
            {/* Federation Results */}
            <div className="grid md:grid-cols-3 gap-4 mb-8">
              {Object.entries(hetResults).map(([fed, metrics]) => (
                <Card key={fed}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center justify-between">
                      {fed.replace("_", " ").toUpperCase()}
                      <Badge variant="secondary">{metrics.model}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-primary mb-2">
                      {(metrics.accuracy * 100).toFixed(1)}%
                    </p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>Precision: {(metrics.precision * 100).toFixed(1)}%</div>
                      <div>Recall: {(metrics.recall * 100).toFixed(1)}%</div>
                      <div>F1: {(metrics.f1 * 100).toFixed(1)}%</div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Heterogeneous Bar Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Heterogeneous MetaFed Results
                </CardTitle>
                <CardDescription>CNN, RNN, and Vision Transformer across federations</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={hetBarData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip formatter={(value: number) => (value * 100).toFixed(1) + "%"} />
                    <Legend />
                    <Bar dataKey="Fed 0 (CNN)" fill="#3b82f6" />
                    <Bar dataKey="Fed 1 (RNN)" fill="#f97316" />
                    <Bar dataKey="Fed 2 (ViT)" fill="#22c55e" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}
