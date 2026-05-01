const root = import.meta.dir;
const port = Number(Bun.env.PORT ?? 3000);

function typeFor(pathname: string) {
  if (pathname.endsWith(".html")) return "text/html; charset=utf-8";
  if (pathname.endsWith(".css")) return "text/css; charset=utf-8";
  if (pathname.endsWith(".js")) return "text/javascript; charset=utf-8";
  return "application/octet-stream";
}

Bun.serve({
  port,
  hostname: "127.0.0.1",
  async fetch(request) {
    const url = new URL(request.url);
    const pathname = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);

    if (pathname.includes("..")) {
      return new Response("Bad request", { status: 400 });
    }

    const file = Bun.file(`${root}${pathname}`);
    if (!(await file.exists())) {
      return new Response("Not found", { status: 404 });
    }

    return new Response(file, {
      headers: { "content-type": typeFor(pathname) },
    });
  },
});

console.log(`http://localhost:${port}`);
