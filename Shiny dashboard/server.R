library(tidyverse)
library(reactable)
library(tidytext)

test_data <- read.csv('C:/Users/andreas.soteriades/Documents/git_projects/positive_about_change_text_mining/y_pred_and_x_test.csv')
accuracy_per_class <- read.csv('C:/Users/andreas.soteriades/Documents/git_projects/positive_about_change_text_mining/accuracy_per_class.csv')

# Define a server for the Shiny app
function(input, output) {
  
  output$pedictedLabels <- renderReactable({
    
    feedback_col_new_name <- paste0(
      "Feedback that model predicted as ", "\"", input$pred, "\""
    )
    
    reactable(
      test_data %>%
      filter(pred == input$pred) %>%
      select(improve),
      columns = list(improve = colDef(name = feedback_col_new_name)),
      #rownames = TRUE,
      searchable = TRUE,
      sortable = FALSE,
      defaultPageSize = 100,
      language = reactableLang(
        searchPlaceholder = "Search for a word..."),
    )
  })
  
  output$modelAccuracyBox <- renderText({
    accuracy_score <- accuracy_per_class %>%
      filter(class == input$pred) %>%
      select(accuracy) %>%
      mutate(accuracy = round(accuracy * 100)) %>%
      pull
    
    paste0("NOTE: Model accuracy for this label is ", accuracy_score, "%. 
           This means that in 100 feedback records, ", accuracy_score, 
           "  are predicted correctly.")
  })
  
  output$tfidf_bars <- renderPlot({
    test_data %>%
      unnest_tokens(word, improve) %>%
      count(pred, word, sort = TRUE) %>%
      bind_tf_idf(word, pred, n) %>%
      arrange(desc(tf_idf)) %>%
      anti_join(stop_words, by = c("word" = "word")) %>% # Do this because some stop words make it through the TF-IDF filtering that happens below.
      as_tibble %>%
      group_by(pred) %>%
      slice_max(tf_idf, n = 15) %>%
      ungroup() %>%
      filter(pred == input$pred) %>%
      ggplot(aes(tf_idf, reorder(word, tf_idf))) +
      geom_col() +
      #labs(x = "TF-IDF*", y = NULL) + 
      labs(x = "TF-IDF*", y = NULL, 
           title = paste0("Most frequent words in feedback text that is about\n", 
                          "\"", input$pred, "\"")) +
      theme_bw() +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
      )
  })
  
  output$tfidfExplanation <- renderText({
    HTML(paste0("*TF-IDF stands for 
          <u><a href='https://en.wikipedia.org/wiki/Tf%E2%80%93idf'>
          Term Frequency–Inverse Document Frequency</a></u>.
          It is a standard way of calculating the frequency (i.e. importance) 
          of a word in the given text. It is a little more sophisticated than
          standard frequency as it adjusts for words that appear too frequently
          in the text. For example, stop words like ", "\"", "a", "\"", " and ",
          "\"", "the", "\"", " are very frequent but uniformative of 
          the cotext of the text."))
  })
}