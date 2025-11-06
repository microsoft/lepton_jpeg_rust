use std::net::SocketAddr;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use lepton_jpeg::{EnabledFeatures, LeptonThreadPool, Metrics, decode_lepton, encode_lepton};
use std::io::Cursor;

struct TokioThreadPool {
}

impl LeptonThreadPool for TokioThreadPool {
    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        tokio::task::spawn_blocking(move || {
            f();
        });
    }
}
async fn compress_response(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    println!("url {}", req.uri().path());

    match (req.method(), req.uri().path()) {
        (&Method::POST, "/api/compress") =>
        {
            let thread_pool = TokioThreadPool{};
            do_work(req, |r,w| encode_lepton(r, w, &EnabledFeatures::compat_lepton_vector_write(), &thread_pool)).await
        },
        (&Method::POST, "/api/uncompress") =>
        {
            let thread_pool = TokioThreadPool{};
            do_work(req, |r,w| decode_lepton(r, w, &EnabledFeatures::compat_lepton_vector_read(), &thread_pool)).await
        },        
        _ => Ok(not_found()),
    }
}

async fn do_work<F : Fn(&mut Cursor<&Bytes>, &mut Cursor<&mut Vec<u8>>) -> Result<Metrics,lepton_jpeg::LeptonError>> (req: Request<hyper::body::Incoming>, f : F) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let b = req.collect().await?.to_bytes();

    println!("body len {}", b.len());
           
    let mut output_data: Vec<u8> = Vec::new();
    {
        let mut reader = Cursor::new(&b);
        let mut writer = Cursor::new(&mut output_data);

        if let Err(e) = f(&mut reader, &mut writer)
        {
            let error_message = format!("Lepton processing error: {}", e);
            
            let bytes = Bytes::from(error_message.as_bytes().to_vec());

            return Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(bytes))
                .unwrap())

        }

    }

    let b = Bytes::from(output_data);

    let response : Response<Full<Bytes>> = Response::builder()
        .status(200)
        .header("Content-Type", "application/octet-stream")
        .body(Full::new(b))
        .unwrap();

    Ok(response)
}

/// HTTP status code 404
fn not_found() -> Response<Full<Bytes>> {
    static NOTFOUND: &[u8] = b"Not Found";
    
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Full::new(NOTFOUND.into()))
        .unwrap()
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {

    let port = std::env::var("FUNCTIONS_CUSTOMHANDLER_PORT").unwrap_or("1337".into());
    let addr = format!("127.0.0.1:{}", port);

    let addr: SocketAddr = addr.parse().unwrap();

    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(compress_response))
                .await
            {
                println!("Failed to serve connection: {:?}", err);
            }
        });
    }
}