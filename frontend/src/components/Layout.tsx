import { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { 
  Home, Database, BarChart3, Users, Upload, GitCompare, Code
} from "lucide-react";

const navItems = [
  { path: "/", label: "Home", icon: Home },
  { path: "/dataset", label: "Dataset", icon: Database },
  { path: "/evaluation", label: "Evaluation Metrics", icon: BarChart3 },
  { path: "/image-upload", label: "Image Upload", icon: Upload },
  { path: "/comparison", label: "Comparison", icon: GitCompare },
];

export default function Layout({ children }: { children: ReactNode }) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gradient-to-r from-primary to-primary/80 text-primary-foreground shadow-lg">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex flex-col">
              <h1 className="text-xl font-bold">MetaFed</h1>
              <span className="text-xs opacity-80">Heterogeneous Federated Learning</span>
            </Link>
            
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={cn(
                      "px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2",
                      isActive 
                        ? "bg-white/20 text-white" 
                        : "text-white/80 hover:bg-white/10 hover:text-white"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                );
              })}
              {/* Backend page removed; keep nav concise */}
            </nav>
          </div>
        </div>
      </header>

      {/* Mobile Navigation */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t z-50">
        <div className="flex justify-around py-2">
          {navItems.slice(0, 5).map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "flex flex-col items-center p-2 text-xs",
                  isActive ? "text-primary" : "text-muted-foreground"
                )}
              >
                <Icon className="h-5 w-5 mb-1" />
                {item.label.split(" ")[0]}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Main Content */}
      <main className="pb-20 md:pb-0">{children}</main>

      {/* Footer */}
      <footer className="bg-muted/50 border-t py-8 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>MetaFed Healthcare Project - Heterogeneous Federated Learning</p>
          <p className="mt-1">B.Tech 3rd Year Academic & Research Project</p>
        </div>
      </footer>
    </div>
  );
}
