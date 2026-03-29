import Layout from "@/components/Layout";
import {useEffect} from "react";
import {useNavigate} from "react-router-dom";

export default function PersonalizedModels() {
  const navigate = useNavigate();
  useEffect(() => { navigate('/', { replace: true }); }, [navigate]);
  return null;
}
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Cpu, Network, Layers } from "lucide-react";

const models = [
  {
    id: "cnn",
    name: "CNN",
    fullName: "Convolutional Neural Network",
    federation: "Federation 0",
    icon: Cpu,
    color: "bg-blue-500",
    accuracy: 89.1,
    description: "1D CNN optimized for time-series sensor data from wearable devices",
    architecture: ["Conv1D (64 filters)", "BatchNorm + ReLU", "MaxPool", "Conv1D (128 filters)", "Global Avg Pool", "FC (128) → FC (12)"],
    params: "~150K parameters",
  },
  {
    id: "rnn",
    name: "RNN (LSTM)",
    fullName: "Long Short-Term Memory",
    federation: "Federation 1",
    icon: Network,
    color: "bg-orange-500",
    accuracy: 86.7,
    description: "Bidirectional LSTM capturing temporal dependencies in activity sequences",
    architecture: ["LSTM (128 hidden)", "2 Layers", "Bidirectional", "Attention", "FC (128) → FC (12)"],
    params: "~200K parameters",
  },
  {
    id: "vit",
    name: "Vision Transformer",
    fullName: "Tiny ViT for Time-Series",
    federation: "Federation 2",
    icon: Layers,
    color: "bg-green-500",
    accuracy: 85.4,
    description: "Transformer encoder with patch embeddings for time-series classification",
    architecture: ["Patch Embedding (10)", "Position Encoding", "Transformer Encoder x2", "CLS Token", "FC (128) → FC (12)"],
    params: "~250K parameters",
  },
];

export default function PersonalizedModels() {
  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Personalized Models</h1>
          <p className="text-muted-foreground">
            Heterogeneous models trained on different federations using MetaFed
          </p>
        </div>

        {/* Info Card */}
        <Card className="max-w-4xl mx-auto mb-8 bg-primary/5 border-primary/20">
          <CardContent className="flex items-center gap-4 p-6">
            <Brain className="h-12 w-12 text-primary" />
            <div>
              <h3 className="font-semibold mb-1">MetaFed Heterogeneous Learning</h3>
              <p className="text-sm text-muted-foreground">
                Each federation uses a different model architecture (CNN, RNN, ViT) while sharing 
                knowledge through cyclic knowledge distillation. This enables personalization 
                while maintaining global knowledge.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Model Cards */}
        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {models.map((model) => {
            const Icon = model.icon;
            return (
              <Card key={model.id} className="overflow-hidden">
                <div className={`h-2 ${model.color}`} />
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${model.color}/10`}>
                        <Icon className={`h-6 w-6 text-${model.color.replace("bg-", "")}`} />
                      </div>
                      <div>
                        <CardTitle className="text-lg">{model.name}</CardTitle>
                        <CardDescription>{model.fullName}</CardDescription>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Badge variant="outline">{model.federation}</Badge>
                  
                  <p className="text-sm text-muted-foreground">{model.description}</p>
                  
                  {/* Accuracy */}
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Accuracy</span>
                      <span className="font-semibold">{model.accuracy}%</span>
                    </div>
                    <Progress value={model.accuracy} className="h-2" />
                  </div>
                  
                  {/* Architecture */}
                  <div>
                    <p className="text-sm font-medium mb-2">Architecture:</p>
                    <div className="space-y-1">
                      {model.architecture.map((layer, idx) => (
                        <div key={idx} className="text-xs bg-muted/50 px-2 py-1 rounded flex items-center gap-2">
                          <span className="text-primary font-mono">{idx + 1}.</span>
                          {layer}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <p className="text-xs text-muted-foreground text-center pt-2 border-t">
                    {model.params}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Knowledge Distillation Explanation */}
        <Card className="max-w-4xl mx-auto mt-8">
          <CardHeader>
            <CardTitle>Cyclic Knowledge Distillation</CardTitle>
            <CardDescription>How heterogeneous models share knowledge</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row items-center justify-center gap-4">
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg flex-1">
                <Cpu className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                <p className="font-semibold">CNN</p>
                <p className="text-xs text-muted-foreground">Teaches spatial features</p>
              </div>
              
              <div className="text-2xl">→</div>
              
              <div className="text-center p-4 bg-orange-50 dark:bg-orange-950 rounded-lg flex-1">
                <Network className="h-8 w-8 mx-auto mb-2 text-orange-500" />
                <p className="font-semibold">RNN</p>
                <p className="text-xs text-muted-foreground">Learns temporal patterns</p>
              </div>
              
              <div className="text-2xl">→</div>
              
              <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg flex-1">
                <Layers className="h-8 w-8 mx-auto mb-2 text-green-500" />
                <p className="font-semibold">ViT</p>
                <p className="text-xs text-muted-foreground">Global attention</p>
              </div>
              
              <div className="text-2xl">↻</div>
            </div>
            <p className="text-sm text-muted-foreground text-center mt-4">
              Knowledge flows cyclically between models, with each model learning from the previous one
              using feature-level knowledge distillation.
            </p>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
