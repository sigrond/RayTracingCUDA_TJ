#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mainFunction.h"
#include "omp.h"

#define NAME ".socket"


int main()
{
	int sock, msgsock, rval;
	struct sockaddr_un server;
	char buf[1024];
	int dummy;


	sock = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock < 0) {
		perror("opening stream socket");
		exit(1);
	}
	server.sun_family = AF_UNIX;
	strcpy(server.sun_path, NAME);
	unlink(NAME);
	if (bind(sock, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
		perror("binding stream socket");
		exit(1);
	}
	//printf("Socket has name %s\n", server.sun_path);
	listen(sock, 5);
	for (;;) {
		msgsock = accept(sock, 0, 0);
		if (msgsock == -1)
			perror("accept");
		else do {
			bzero(buf, sizeof(buf));
			if ((rval = read(msgsock, buf, 1024)) < 0)
				perror("reading stream message");
			else if (rval == 0)
				printf("Ending connection\n");
			else{
				double init = omp_get_wtime();
				mainFunction();
				printf("%f LACZNY CZAS\n", omp_get_wtime() - init);
				if (write(msgsock, &dummy, sizeof(int)) < 0 )
					perror("error writing stream socket");
				}
		} while (rval > 0);
		close(msgsock);
		//break; //TODO: WAZNE TO BLOKUJE TERAZ SERWER (04.04.13 by szmigacz)
	}
	close(sock);
	unlink(NAME);
	return 0;
}


