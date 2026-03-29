import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import Layout from "@/components/Layout";
import { 
  CheckCircle, XCircle, Loader2, Database, Play, ArrowRight,
  Brain, Network, Layers, BarChart3
} from "lucide-react";

export default function Home() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [preprocessed, setPreprocessed] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const data = await api.health();
      setStatus(data);
      setBackendOnline(true);
      setPreprocessed(data.preprocessing_done || false);
    } catch (err) {
      setBackendOnline(false);
    }
  };

  const handlePreprocess = async () => {
    setLoading(true);
    try {
      await api.preprocess();
      setPreprocessed(true);
      toast({ title: "Success", description: "Dataset preprocessed successfully!" });
    } catch (err) {
      toast({ title: "Error", description: "Preprocessing failed. Is backend running?", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const features = [
    { icon: Network, title: "3 Federations", desc: "Decentralized training across distributed data centers" },
    { icon: Layers, title: "Multiple Algorithms", desc: "FedAvg, FedProx, FedBN, MetaFed comparison" },
    { icon: Brain, title: "Heterogeneous Models", desc: "CNN + RNN + Vision Transformer" },
    { icon: BarChart3, title: "Comprehensive Metrics", desc: "Accuracy, Precision, Recall, F1-Score" },
  ];

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <Badge className="mb-4" variant="secondary">Academic Research Project</Badge>
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            MetaFed: Heterogeneous<br />Federated Learning
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
            Federated Learning Among Federations with Cyclic Knowledge Distillation 
            for Multi-Domain Activity Recognition
          </p>
          
          <div className="flex flex-wrap justify-center gap-4">
            <Button 
              size="lg" 
              onClick={handlePreprocess} 
              disabled={loading || preprocessed}
              className="min-w-[200px]"
            >
              {loading ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Preprocessing...</>
              ) : preprocessed ? (
                <><CheckCircle className="mr-2 h-4 w-4" /> Dataset Ready</>
              ) : (
                <><Database className="mr-2 h-4 w-4" /> Preprocess Dataset</>
              )}
            </Button>
            
            <Button 
              size="lg" 
              variant="outline"
              onClick={() => navigate("/evaluation")}
              disabled={!preprocessed}
            >
              <Play className="mr-2 h-4 w-4" />
              Run Experiments
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Status Cards */}
        <div className="grid md:grid-cols-2 gap-4 max-w-2xl mx-auto mb-12">
          <Card>
            <CardContent className="flex items-center gap-4 p-4">
              {backendOnline ? (
                <CheckCircle className="h-8 w-8 text-green-500" />
              ) : (
                <XCircle className="h-8 w-8 text-destructive" />
              )}
              <div>
                <p className="font-semibold">Backend Status</p>
                <p className="text-sm text-muted-foreground">
                  {backendOnline ? "Running on localhost:5000" : "Not connected"}
                </p>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="flex items-center gap-4 p-4">
              {preprocessed ? (
                <CheckCircle className="h-8 w-8 text-green-500" />
              ) : (
                <XCircle className="h-8 w-8 text-yellow-500" />
              )}
              <div>
                <p className="font-semibold">Dataset Status</p>
                <p className="text-sm text-muted-foreground">
                  {preprocessed ? "PAMAP2 Preprocessed" : "Not preprocessed yet"}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {features.map((feature, idx) => {
            const Icon = feature.icon;
            return (
              <Card key={idx} className="text-center">
                <CardContent className="pt-6">
                  <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.desc}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Pipeline Diagram */}
        <Card className="max-w-4xl mx-auto">
          <CardHeader>
            <CardTitle>MetaFed Training Pipeline</CardTitle>
            <CardDescription>Two-stage federated learning approach</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center justify-center gap-4">
              <div className="flex-1 p-6 bg-primary/5 rounded-lg text-center border-2 border-primary/20">
                <h4 className="font-semibold text-primary mb-2">Stage I</h4>
                <p className="text-sm">Common Knowledge Accumulation</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Cyclic knowledge distillation across federations
                </p>
              </div>
              
              <ArrowRight className="h-8 w-8 text-primary hidden md:block" />
              <ArrowRight className="h-8 w-8 text-primary rotate-90 md:hidden" />
              
              <div className="flex-1 p-6 bg-secondary/50 rounded-lg text-center border-2 border-secondary">
                <h4 className="font-semibold text-secondary-foreground mb-2">Stage II</h4>
                <p className="text-sm">Personalization</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Fine-tuning on local federation data
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
