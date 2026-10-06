namespace NNtrain.Gui;

public sealed record ChatTurn(string Role, string Content, string? AssistantPrefix = null, ChatImage? Image = null);
