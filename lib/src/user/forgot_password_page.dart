import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/components/custom_textfield.dart';
import 'package:railway/src/utils/constants.dart';

class ForgotPasswordPage extends StatefulWidget {
  const ForgotPasswordPage({super.key});

  @override
  State<ForgotPasswordPage> createState() => _ForgotPasswordPageState();
}

class _ForgotPasswordPageState extends State<ForgotPasswordPage> {
  final _emailController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  Future passwordReset() async {
    try {
      await FirebaseAuth.instance
          .sendPasswordResetEmail(email: _emailController.text.trim());
      // ignore: use_build_context_synchronously
      if (!context.mounted) return;
      showAlertDialogueBox(
          context, "Password reset link sent! Check your email");
    } on FirebaseAuthException catch (e) {
      showAlertDialogueBox(context, e.message.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: primaryColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
            color: bgColor
        ),
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 100,),
            // app icon
            Padding(
              padding: const EdgeInsets.only(top: 60, bottom: 20),
              child: Image.asset(
                "assets/images/icons/logo.png",
                width: 150,
              ),
            ),

            // login text
            Padding(
              padding: const EdgeInsets.only(left: 15.0, right: 15.0, top: 50.0, bottom: 15.0),
              child: Text(
                "Enter your email and we will send you a password reset link",
                style: TextStyle(
                    fontSize: 18,
                    color: primaryColor,
                ),
                textAlign: TextAlign.center,
              ),
            ),

            Padding(
              padding: const EdgeInsets.all(8.0),
              child: LoginTextField(
                controller: _emailController,
                hintText: "Email",
                textInputAction: TextInputAction.next,
                keyboardType: TextInputType.emailAddress,
                prefix: const Icon(Icons.email_rounded),
              ),
            ),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 50.0),
              child: MaterialButton(
                onPressed: () {
                  passwordReset();
                },
                color: primaryColor,
                textColor: Colors.white,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                    side: BorderSide(color: primaryColor)),
                child: const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 5, vertical: 15),
                  child: Text("Reset Password"),
                ),
              ),
            )
          ],
        ),
          ),
        ),
      ),
    );
  }
}
