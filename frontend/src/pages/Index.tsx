import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { Code, Server, Layout, ArrowRight } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            MetaFed Healthcare Project
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Heterogeneous Federated Learning for Activity Recognition using PAMAP2 Dataset
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          <Card className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Server className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>Step 1: Backend Setup</CardTitle>
              </div>
              <CardDescription>
                Python Flask API with ML algorithms
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground space-y-2 mb-4">
                <li>• PAMAP2 data preprocessing</li>
                <li>• CNN, RNN, Vision Transformer models</li>
                <li>• FedAvg, FedBN, FedProx, MetaFed algorithms</li>
                <li>• Heterogeneous MetaFed extension</li>
                <li>• REST API endpoints</li>
              </ul>
              <Link to="/backend-code">
                <Button className="w-full">
                  <Code className="mr-2 h-4 w-4" />
                  View Python Code
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="border-2 border-dashed opacity-60">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-muted rounded-lg">
                  <Layout className="h-6 w-6 text-muted-foreground" />
                </div>
                <CardTitle>Step 2: Frontend (Coming Next)</CardTitle>
              </div>
              <CardDescription>
                React dashboard to visualize results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground space-y-2 mb-4">
                <li>• Home page with project overview</li>
                <li>• Dataset visualization</li>
                <li>• Real-time training dashboard</li>
                <li>• Evaluation metrics & charts</li>
                <li>• Algorithm comparison</li>
              </ul>
              <Button variant="outline" className="w-full" disabled>
                Available after backend setup
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-muted-foreground">
            First, set up the Python backend locally, then we'll build the React frontend to connect to it.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
