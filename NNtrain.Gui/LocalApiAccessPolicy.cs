using System.Net;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http;

namespace NNtrain.Gui;

/// <summary>Checks the loopback-only contract before admitting any endpoint request.</summary>
internal sealed class LocalApiAccessPolicy : IDisposable
{
    private readonly byte[] _authenticationBytes;
    private readonly SemaphoreSlim _admission = new(4, 4);

    internal LocalApiAccessPolicy(string authenticationToken)
        => _authenticationBytes = Encoding.ASCII.GetBytes("Bearer " + authenticationToken);

    internal int? RejectionStatus(HttpContext context)
    {
        string host = context.Request.Host.Host;
        if (context.Connection.RemoteIpAddress is not { } remote || !IPAddress.IsLoopback(remote)
            || !(host.Equals("localhost", StringComparison.OrdinalIgnoreCase) || host == "127.0.0.1")
            || context.Request.Headers.ContainsKey("Origin"))
            return StatusCodes.Status403Forbidden;
        byte[] supplied = Encoding.ASCII.GetBytes(context.Request.Headers.Authorization.ToString());
        if (!CryptographicOperations.FixedTimeEquals(supplied, _authenticationBytes))
            return StatusCodes.Status401Unauthorized;
        if (HttpMethods.IsPost(context.Request.Method) && !context.Request.HasJsonContentType())
            return StatusCodes.Status415UnsupportedMediaType;
        return null;
    }

    internal async Task InvokeAsync(HttpContext context, RequestDelegate next)
    {
        if (RejectionStatus(context) is { } status)
        {
            context.Response.StatusCode = status;
            return;
        }
        if (!await _admission.WaitAsync(0, context.RequestAborted).ConfigureAwait(false))
        {
            context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
            context.Response.Headers.RetryAfter = "1";
            return;
        }
        try { await next(context).ConfigureAwait(false); }
        finally { _admission.Release(); }
    }

    public void Dispose() => _admission.Dispose();
}
