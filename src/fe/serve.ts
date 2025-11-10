// Simple static file server for the frontend with TypeScript transpilation
const transpiler = new Bun.Transpiler({
  loader: "ts",
  target: "browser",
});

const server = Bun.serve({
  port: 8001,

  async fetch(req: Request): Promise<Response> {
    const url: URL = new URL(req.url);
    let filePath: string = url.pathname;

    // Default to index.html
    if (filePath === "/") {
      filePath = "/index.html";
    }

    try {
      const fullPath = `.${filePath}`;

      // Handle TypeScript files - transpile them to JavaScript
      if (filePath.endsWith(".ts")) {
        const file = Bun.file(fullPath);
        const exists = await file.exists();

        if (!exists) {
          return new Response("Not Found", { status: 404 });
        }

        const code = await file.text();
        const transpiled = await transpiler.transform(code);

        return new Response(transpiled, {
          headers: {
            "Content-Type": "application/javascript",
          },
        });
      }

      // Serve other files normally
      const file = Bun.file(fullPath);
      return new Response(file);
    } catch (error) {
      console.error("Error serving file:", error);
      return new Response("Not Found", { status: 404 });
    }
  },
});

console.log(`🌐 Frontend server running on http://localhost:${server.port}`);
console.log(`   Open http://localhost:${server.port} in your browser`);
