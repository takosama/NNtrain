namespace NNtrain.Gui.Tests;

// Can be compiled with the installed SDK and existing xUnit binaries when package restore is unavailable.
public static class OfflineRunner
{
    public static int Main()
    {
        var tests = new AudioGuiTests();
        try
        {
            if (System.Environment.GetEnvironmentVariable("NNTRAIN_ASR_LOAD_GUI_TEST") == "1")
            {
                AsrLoadingGuiChecks.Run(); return 0;
            }
            if (System.Environment.GetEnvironmentVariable("NNTRAIN_LOCAL_API_SECURITY_TEST") == "1")
            {
                new LocalApiSecurityTests().ManagementRequiresSessionAuthenticationAndRejectsBrowserAndNetworkPaths().GetAwaiter().GetResult();
                System.Console.WriteLine("Local API security CPU checks passed. No model, GPU or microphone started.");
                return 0;
            }
            if (System.Environment.GetEnvironmentVariable("NNTRAIN_ASR_REAL_COMPOSER_TEST") == "1")
            {
                RealComposerChecks.Run(); return 0;
            }
            if (System.Environment.GetEnvironmentVariable("NNTRAIN_ASR_REALTIME_GUI_TEST") == "1")
            {
                RealtimeAudioGuiChecks.Run(); return 0;
            }
            if (System.Environment.GetEnvironmentVariable("NNTRAIN_ASR_REAL_GUI_TEST") == "1")
            {
                RealAudioGuiChecks.Run();
                return 0;
            }
            tests.EditedTranscriptMovesIntoExistingChatInputWithoutRecording();
            tests.CancelRejectsQueuedPartialAndFinalFromOldUtterance();
            tests.ActiveRecognitionCannotReplaceChatDraft();
            tests.EditedRecognitionUsesExistingLocalChatTransport();
            tests.LivePartialUpdatesBothTextBoxesAndCancelClearsCurrentUtterance();
            tests.FinalizationRejectsQueuedPartialsAndPreservesSubsequentEdits();
            tests.AudioInsertionKeepsSelectedTextAndSuffix();
            tests.NavigatingTabsKeepsLiveTextAndModelSwitchRestoresDraft();
            tests.IntegratedLayoutFitsSmallAndDefaultWindowSizes();
            tests.JapaneseModelSelectorChangesBackendFolderAndReleasesPreviousModel();
            tests.FinalResultReachesComposerBeforeEditingIsReenabled();
            System.Console.WriteLine("Audio GUI checks passed: 11 (chat integration, final dispatch, model selection, layout and mock transport). No server or microphone started.");
            return 0;
        }
        catch (System.Exception ex) { System.Console.Error.WriteLine(ex); return 1; }
    }
}
