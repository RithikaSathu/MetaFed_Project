import Layout from "@/components/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { DATASET_INFO } from "@/lib/api";
import { Database, Users, Activity, Cpu, Clock, Layers } from "lucide-react";

export default function Dataset() {
  const stats = [
    { icon: Users, label: "Subjects", value: DATASET_INFO.subjects },
    { icon: Activity, label: "Activities", value: DATASET_INFO.activities },
    { icon: Cpu, label: "IMU Channels", value: DATASET_INFO.imuChannels },
    { icon: Clock, label: "Sampling Rate", value: DATASET_INFO.samplingRate },
    { icon: Layers, label: "Window Size", value: DATASET_INFO.windowSize },
    { icon: Database, label: "Federations", value: DATASET_INFO.federations },
  ];

  const federationDetails = [
    { id: "Fed 0", subjects: "Subject 1-3", model: "CNN", color: "bg-blue-500" },
    { id: "Fed 1", subjects: "Subject 4-6", model: "RNN", color: "bg-orange-500" },
    { id: "Fed 2", subjects: "Subject 7-9", model: "ViT", color: "bg-green-500" },
  ];

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Dataset</h1>
          <p className="text-muted-foreground">PAMAP2 Physical Activity Monitoring Dataset</p>
        </div>

        {/* Dataset Selector */}
        <Card className="max-w-md mx-auto mb-8">
          <CardHeader>
            <CardTitle className="text-lg">Select Dataset</CardTitle>
          </CardHeader>
          <CardContent>
            <Select defaultValue="pamap2">
              <SelectTrigger>
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pamap2">PAMAP2 - Physical Activity Monitoring</SelectItem>
              </SelectContent>
            </Select>
          </CardContent>
        </Card>

        {/* Dataset Info */}
        <Card className="max-w-4xl mx-auto mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              {DATASET_INFO.name}
            </CardTitle>
            <CardDescription>{DATASET_INFO.fullName}</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">{DATASET_INFO.description}</p>
            
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {stats.map((stat, idx) => {
                const Icon = stat.icon;
                return (
                  <div key={idx} className="text-center p-4 bg-muted/50 rounded-lg">
                    <Icon className="h-5 w-5 mx-auto mb-2 text-primary" />
                    <p className="text-2xl font-bold">{stat.value}</p>
                    <p className="text-xs text-muted-foreground">{stat.label}</p>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        {/* Activity Labels */}
        <Card className="max-w-4xl mx-auto mb-8">
          <CardHeader>
            <CardTitle>Activity Classes</CardTitle>
            <CardDescription>12 physical activities recognized by the model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {DATASET_INFO.activityLabels.map((activity, idx) => (
                <Badge key={idx} variant="outline" className="text-sm">
                  {idx}: {activity}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Federation Distribution */}
        <Card className="max-w-4xl mx-auto">
          <CardHeader>
            <CardTitle>Federation Distribution</CardTitle>
            <CardDescription>How subjects are distributed across federations</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Federation</TableHead>
                  <TableHead>Subjects</TableHead>
                  <TableHead>Model (Heterogeneous)</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {federationDetails.map((fed) => (
                  <TableRow key={fed.id}>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${fed.color}`} />
                        {fed.id}
                      </div>
                    </TableCell>
                    <TableCell>{fed.subjects}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{fed.model}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-green-600">Active</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}
