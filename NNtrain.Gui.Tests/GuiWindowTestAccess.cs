using System.Reflection;

namespace NNtrain.Gui.Tests;

/// <summary>Shared access to window characterization state after service extraction.</summary>
internal static class GuiWindowTestAccess
{
    private const BindingFlags InstanceFields = BindingFlags.Instance | BindingFlags.NonPublic;

    internal static object? Get(MainWindow window, string name)
    {
        (object owner, FieldInfo field) = Resolve(window, name);
        return field.GetValue(owner);
    }

    internal static void Set(MainWindow window, string name, object? value)
    {
        (object owner, FieldInfo field) = Resolve(window, name);
        field.SetValue(owner, value);
    }

    private static (object Owner, FieldInfo Field) Resolve(MainWindow window, string name)
    {
        // Characterization checks still describe the same connection and owned process;
        // their storage now belongs to the transport/process services.
        (string? service, string member) = name switch
        {
            "_client" => ("_api", "_client"),
            "_serverProcess" => ("_server", "_process"),
            "_serverReady" => ("_server", "<IsReady>k__BackingField"),
            _ => (null, name)
        };
        object owner = service is null ? window : typeof(MainWindow).GetField(service, InstanceFields)!.GetValue(window)!;
        return (owner, owner.GetType().GetField(member, InstanceFields)!);
    }
}
